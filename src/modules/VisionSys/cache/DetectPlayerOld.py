def DetectPlayers(self, img, timestamp, dbg=False, isT=False, hsv_img=None):
        """
        
        Detecta robôs na imagem com distinção por dominância de cor
        (comparando proporção de área entre cor de aliado e inimigo).
        Em modo debug, gera máscaras binárias de aliados e de todos os robôs.
        """
        # 1. Usa o HSV Global
        if hsv_img is not None:
            imgHSV = hsv_img
        else:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        debug = dbg
        # Reset de contadores e status
        self.playersCount = self.enemiesCount = self.alliesCount = 0
        enemies_count = 0
        for bot in (*self.enemyTeam, *self.allyTeam):
            bot.setStatus(False)

        self.binaryPlayers = cv2.inRange(imgHSV, self.objectsDarkColor, self.objectsLightColor)
       
        # 2. Remoção cirúrgica da bola na máscara de objetos
        if self.ball.status:
            # Coordenadas inteiras da bola na imagem reduzida
            xb, yb = int(self.ball.xb), int(self.ball.yb)
            
            # Define o raio da janela (ex: r=5 para uma janela 10x10)
            r = 5 
            
            # Obtém as dimensões para evitar erro de índice fora da imagem
            h, w = self.binaryPlayers.shape[:2]
            
            # Calcula os limites com clamp (garante que fiquem dentro da imagem)
            y1, y2 = max(0, yb - r), min(h, yb + r)
            x1, x2 = max(0, xb - r), min(w, xb + r)
            
            # OPERAÇÃO DIRETA: Zera os pixels onde a bola está
            self.binaryPlayers[y1:y2, x1:x2] = 0


        # Filtragem morfológica (reutiliza elementos estruturais em cache)
        ellipse5 = self.struct_ellipse5
        rect11 = self.struct_rect11
        self.binaryPlayers = cv2.erode(self.binaryPlayers, ellipse5, iterations=1)
        self.binaryPlayers = cv2.morphologyEx(self.binaryPlayers, cv2.MORPH_CLOSE, rect11)

        # Reconhecimento de objetos
        self.binaryPlayers, players = self.DetectSquares(self.binaryPlayers)

        # Pré-cálculos
        winSize = int(18 * self.prop_px_cm)
        playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
        mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm

        # Flags de controle
        AgoalFlag = Aatk1Flag = Aatk2Flag = False

        # Inicializa máscaras de debug se necessário
        if debug:
            binaryAllies = np.zeros(img.shape[:2], dtype=np.uint8)
            binaryAllTeam = np.zeros(img.shape[:2], dtype=np.uint8)

        for i, currentPlayer in enumerate(players):
            # Coordenada do centro do robô na imagem reduzida, já detectada.
            (xi, yi), ri = cv2.minEnclosingCircle(currentPlayer)
            
            if debug:
                cv2.circle(self.frameResult, (int(xi), int(yi)), int(ri) + 5, (0, 255, 0), 2)       
            if not (0.2 * playerRadius < ri < 2 * playerRadius and self.playersCount < 6):
                continue


            # recorta área de interesse
            half_win = winSize // 2
            x1, y1 = max(0, int(xi - half_win)), max(0, int(yi - half_win))
            x2, y2 = min(img.shape[1], int(xi + half_win)), min(img.shape[0], int(yi + half_win))
            windowActual = img[y1:y2, x1:x2]

            # converte a região para HSV
            hsv = imgHSV[y1:y2, x1:x2]

            # Máscaras de cor
            mask_ally = self.MaskInRange(hsv, self.ally_lower_bound, self.ally_upper_bound)
            mask_enemy = self.MaskInRange(hsv, self.enemy_lower_bound, self.enemy_upper_bound)

            # Áreas detectadas
            ally_area = cv2.countNonZero(mask_ally)
            enemy_area = cv2.countNonZero(mask_enemy)
            total_area = max(ally_area + enemy_area, 1)

            # Proporções relativas
            ally_ratio = ally_area / total_area
            enemy_ratio = enemy_area / total_area

            # Decide dominância
            if ally_ratio > 0.45:
                team_type = "ally"
            elif enemy_ratio > 0.45:
                team_type = "enemy"
            else:
                team_type = "uncertain"
                #if debug:
                #    print(f"[DetectPlayers] team_type uncertain for window at ({xi:.1f}, {yi:.1f})")

            xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([xi, yi])))
            rcm = 5.30

            # --- INIMIGOS ---
            if team_type == "enemy" and self.enemiesCount < 3:
                contour = max(cv2.findContours(mask_enemy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                            key=cv2.contourArea, default=None)
                if contour is not None:
                    (x_m, y_m), rc = cv2.minEnclosingCircle(contour)            

                    # Transformar as coordenadas da janela para as coordenadas reais (somando o extremo novamente)
                    x_m = x_m+x1
                    y_m = y_m+y1

                    if rc >= 0.75* mainColorRadius:
                        # Direção do robô nas coordenadas da imagem, apenas transformando corretamente
                        # como as coordenadas da imagem tem y negativo como padrão, inverte o sinal dele
                        direction = np.array([xi,-yi]) - np.array([x_m,-y_m]) 

                        # Normalizando
                        modDir = np.linalg.norm(direction)
                        if modDir > 1e-6:
                            direction = direction/modDir

                        # Adquirindo as cores principais do robô
                        C_p, C_s, _ = self.GetCentersColors(xi,yi, x_m, y_m)

                        # Cor principal do inimigo
                        Color_p = self.GetHsvMean(imgHSV, C_p[0],C_p[1])

                        # Cor secundária do inimigo
                        Color_s = self.GetHsvMean(imgHSV, C_s[0],C_s[1])

                        # Adicionando o robô à estrutura de arvore
                        self.colorTree.add_robot(
                            ID_Team.TEAM_ENEMY,
                            enemies_count,
                            self.enemyColor, 
                            Color_p,
                            Color_s
                        )

                        # Pegar quais são essas cores 
                        bot = self.enemyTeam[self.enemiesCount]
                        
                        # ---------------------------------------------------------
                        # Persistência do filtro de Kalman:
                        # Se o filtro ainda não foi inicializado, inicialize com setPosition.
                        # Caso contrário, apenas atualize o filtro com updatePosition.
                        # ---------------------------------------------------------
                        if not bot.kalman_initialized:
                            bot.setPosition(xcm, ycm, direction, windowActual, time=timestamp)
                        else:
                            bot.updatePosition(xcm, ycm, direction, windowActual, time=timestamp)
                        
                        bot.updtPositionImg(xi, yi, ri)
                        bot.setStatus(True)
                        bot.setRadius(rcm)
                        # Robô inimigo: cor de time correta é enemyColor (não allyColor),
                        # senão matches_robot/colorTree buscariam com a cor principal errada.
                        bot.setColor(colorT=self.enemyColor, colorP=Color_p, colorS=Color_s)
                        self.DrawPlayerVirtual(bot)
                        self.enemiesCount += 1
                        enemies_count +=1

                        if debug:
                            self.DrawPlayerCircle(self.frameResult, bot)

                            # --- C_p ---
                            cx, cy = int(C_p[0]), int(C_p[1])
                            bgr_p = self.Hsv2Bgr(Color_p)

                            cv2.circle(self.frameResult, (cx, cy), 4, (0, 0, 0), -1)   # borda preta
                            cv2.circle(self.frameResult, (cx, cy), 3, bgr_p, -1)       # dentro colorido

                            # --- C_s ---
                            cx2, cy2 = int(C_s[0]), int(C_s[1])
                            bgr_s = self.Hsv2Bgr(Color_s)

                            cv2.circle(self.frameResult, (cx2, cy2), 4, (0, 0, 0), -1)  # borda preta
                            cv2.circle(self.frameResult, (cx2, cy2), 3, bgr_s, -1)      # dentro colorido


            # --- ALIADOS ---
            elif team_type == "ally" and self.alliesCount < 3:
                contour = max(cv2.findContours(mask_ally, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                            key=cv2.contourArea, default=None)
                if contour is not None:
                    (x_m, y_m), rc = cv2.minEnclosingCircle(contour)
                    
                    # Transformar as coordenadas da janela para as coordenadas reais (somando o extremo novamente)
                    x_m = x_m+x1
                    y_m = y_m+y1

                    if rc >= 0.5 * mainColorRadius:
                        ally_checks = [
                            (not AgoalFlag, self.goalAllyColor1, self.goalAllyColor2, ID_Robots.ROBOT_ALLY_GOAL, "Goleiro"),
                            (not Aatk1Flag, self.atk1AllyColor1, self.atk1AllyColor2, ID_Robots.ROBOT_ALLY_1, "Atacante 1"),
                            (not Aatk2Flag, self.atk2AllyColor1, self.atk2AllyColor2, ID_Robots.ROBOT_ALLY_2, "Atacante 2"),
                        ]

                        for flag, c1, c2, bot_id, name in ally_checks:
                            if flag and self.DetectAllyRobot(windowActual, c1, c2):
                                # Direção do robô nas coordenadas da imagem, apenas transformando corretamente
                                # como as coordenadas da imagem tem y negativo como padrão, inverte o sinal dele
                                direction = np.array([xi,-yi]) - np.array([x_m,-y_m]) 

                                # Normalizando
                                modDir = np.linalg.norm(direction)
                                if modDir > 1e-6:
                                    direction = direction/modDir

                                bot = self.allyTeam[bot_id]
                                
                                # ---------------------------------------------------------
                                # Persistência do filtro de Kalman:
                                # Se o filtro ainda não foi inicializado, inicialize com setPosition.
                                # Caso contrário, apenas atualize o filtro com updatePosition.
                                # ---------------------------------------------------------
                                if not bot.kalman_initialized:
                                    bot.setPosition(xcm, ycm, direction, windowActual, time=timestamp)
                                else:
                                    bot.updatePosition(xcm, ycm, direction, windowActual, time=timestamp)
                                
                                bot.updtPositionImg(xi, yi, ri)
                                bot.setStatus(True)
                                bot.setRadius(rcm)
                                bot.setColor(colorT=self.allyColor, colorP=c1, colorS=c2)
                                #if debug:
                                    #print(f"[DetectPlayers][ally] detected {name} at ({x_m:.1f}, {y_m:.1f})")
                                if self.debug: self.DrawPlayerCircle(self.frameResult, bot)
                                self.DrawPlayerVirtual(bot)
                                if bot_id == ID_Robots.ROBOT_ALLY_GOAL:
                                    AgoalFlag = True
                                elif bot_id == ID_Robots.ROBOT_ALLY_1:
                                    Aatk1Flag = True
                                else:
                                    Aatk2Flag = True

                                if debug:
                                    # ponto da cor dominante (x_m, y_m)
                                    cv2.circle(self.frameResult, (int(x_m), int(y_m)), 4, (255, 128, 255), -1) # laranja

                                break

                        self.alliesCount = min(self.alliesCount + 1, 3)

                        # Adiciona aos binários em debug
                        if debug:
                            cv2.drawContours(binaryAllies, [currentPlayer], -1, 255, -1)
                            cv2.drawContours(binaryAllTeam, [currentPlayer], -1, 255, -1)

            self.playersCount += 1

        if debug:
            self.binaryAllies = binaryAllies
            self.binaryAllTeam = binaryAllTeam

        self._countProcess += 1
    