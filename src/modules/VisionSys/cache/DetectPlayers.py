# Substituto do método DetectPlayers original.
# Pressupõe as mesmas imports/atributos já presentes na classe original
# (cv2, np, ID_Team, ID_Robots, self.struct_ellipse5, self.struct_rect11,
# self.objectsDarkColor/objectsLightColor, self.ally_lower_bound/upper_bound,
# self.enemy_lower_bound/upper_bound, self.goalAllyColor1/2,
# self.atk1AllyColor1/2, self.atk2AllyColor1/2, self.colorTree, etc.)
#
# v3 - O QUE MUDOU EM RELAÇÃO À v2 (que causou flickering / perda de detecção):
#
#   A v2 buscava os blobs já filtrando pela máscara de cor de TIME
#   (ally_lower_bound/enemy_lower_bound) antes de testar a forma quadrada.
#   Essa máscara é mais estreita/ruidosa que a genérica (objectsDarkColor/
#   objectsLightColor), então o contorno resultante, depois de dilatado,
#   nem sempre fechava num quadrado limpo de 4 vértices -> is_square
#   reprovava candidatos que antes passavam, causando flickering (inclusive
#   em inimigos, cuja identidade nem depende de cor secundária).
#
#   v3 volta a achar os blobs EXATAMENTE como o código original (uma única
#   máscara genérica -> erode -> close -> DetectSquares), e só entra na
#   lógica anti-colisão quando a janela de análise mostra as duas cores
#   (aliado E inimigo) presentes em quantidade relevante ao mesmo tempo -
#   que é o sintoma real de dois robôs de times diferentes fundidos num
#   blob só. Nesse caso (raro, só em colisão de fato), o blob é dividido em
#   dois sub-candidatos usando as máscaras de cor específicas dentro de uma
#   janela um pouco maior. Fora desse caso, o comportamento é idêntico ao
#   original - sem regressão de recall/estabilidade.
#
#   [FIX #4 - identidade estável de inimigos] mantido: atribuição por
#   vizinho mais próximo em vez de ordem de varredura (self._enemy_last_pos).
#
#   [FIX #5 - gating de plausibilidade] mantido: uma detecção só atualiza um
#   robô se estiver a no máximo MAX_JUMP_CM da última posição conhecida
#   dele.

def DetectPlayers(self, img, timestamp, dbg=False, isT=False, hsv_img=None):
        """
        Detecta robôs na imagem. Pipeline de blob idêntico ao original; a
        diferença só aparece quando uma janela mostra cor de aliado E de
        inimigo em quantidade relevante ao mesmo tempo (colisão) - nesse caso
        o blob é dividido em dois candidatos em vez de descartado ou
        misclassificado.
        """
        # 1. Usa o HSV Global
        if hsv_img is not None:
            imgHSV = hsv_img
        else:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        debug = dbg

        # Cache de últimas posições conhecidas (persiste entre frames).
        if not hasattr(self, "_enemy_last_pos"):
            self._enemy_last_pos = {}   # {slot_idx (0,1,2): (xcm, ycm)}
        if not hasattr(self, "_ally_last_pos"):
            self._ally_last_pos = {}    # {bot_id (ID_Robots): (xcm, ycm)}

        # Reset de contadores e status
        self.playersCount = self.enemiesCount = self.alliesCount = 0
        for bot in (*self.enemyTeam, *self.allyTeam):
            bot.setStatus(False)

        ellipse5 = self.struct_ellipse5
        rect11 = self.struct_rect11

        # --------------------------------------------------------------------
        # Pipeline de blob = ORIGINAL (sem filtro de cor de time aqui).
        # --------------------------------------------------------------------
        self.binaryPlayers = cv2.inRange(imgHSV, self.objectsDarkColor, self.objectsLightColor)

        # Remoção cirúrgica da bola na máscara de objetos
        if self.ball.status:
            xb, yb = int(self.ball.xb), int(self.ball.yb)
            r = 5
            h, w = self.binaryPlayers.shape[:2]
            y1b, y2b = max(0, yb - r), min(h, yb + r)
            x1b, x2b = max(0, xb - r), min(w, xb + r)
            self.binaryPlayers[y1b:y2b, x1b:x2b] = 0

        self.binaryPlayers = cv2.erode(self.binaryPlayers, ellipse5, iterations=1)
        self.binaryPlayers = cv2.morphologyEx(self.binaryPlayers, cv2.MORPH_CLOSE, rect11)
        self.binaryPlayers, players = self.DetectSquares(self.binaryPlayers)

        # Pré-cálculos
        winSize = int(18 * self.prop_px_cm)
        half_win = winSize // 2
        playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
        mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm

        MAX_JUMP_CM = 25.0        # gating de distância (FIX #5)
        MIN_COLOR_PIXELS = 8      # ignora ruído mínimo ao checar mistura de cor
        MIX_RATIO_MIN = 0.25      # cor minoritária precisa ter >= 25% da majoritária pra contar como mistura real
        MERGE_SIZE_FACTOR = 1.15  # blob precisa ser >=15% maior que um robô normal pra suspeitar de fusão
        # As quatro constantes acima são só um ponto de partida - vale calibrar
        # olhando imagens reais de colisão do seu time.

        AgoalFlag = Aatk1Flag = Aatk2Flag = False

        if debug:
            binaryAllies = np.zeros(img.shape[:2], dtype=np.uint8)
            binaryAllTeam = np.zeros(img.shape[:2], dtype=np.uint8)

        ally_candidates = []
        enemy_candidates = []

        for currentPlayer in players:
            (xi, yi), ri = cv2.minEnclosingCircle(currentPlayer)

            if debug:
                cv2.circle(self.frameResult, (int(xi), int(yi)), int(ri) + 5, (0, 255, 0), 2)

            if not (0.2 * playerRadius < ri < 2 * playerRadius and self.playersCount < 6):
                continue
            self.playersCount += 1

            x1, y1 = max(0, int(xi - half_win)), max(0, int(yi - half_win))
            x2, y2 = min(img.shape[1], int(xi + half_win)), min(img.shape[0], int(yi + half_win))
            windowActual = img[y1:y2, x1:x2]
            if windowActual.size == 0:
                continue
            hsv = imgHSV[y1:y2, x1:x2]

            # Máscaras de cor (igual ao original)
            mask_ally = self.MaskInRange(hsv, self.ally_lower_bound, self.ally_upper_bound)
            mask_enemy = self.MaskInRange(hsv, self.enemy_lower_bound, self.enemy_upper_bound)
            ally_area = cv2.countNonZero(mask_ally)
            enemy_area = cv2.countNonZero(mask_enemy)
            total_area = max(ally_area + enemy_area, 1)
            ally_ratio = ally_area / total_area
            enemy_ratio = enemy_area / total_area

            is_mixed = (
                ri > MERGE_SIZE_FACTOR * playerRadius
                and ally_area >= MIN_COLOR_PIXELS and enemy_area >= MIN_COLOR_PIXELS
                and min(ally_area, enemy_area) / max(ally_area, enemy_area) >= MIX_RATIO_MIN
            )

            if is_mixed:
                # ------------------------------------------------------------
                # Sinal real de colisão aliado x inimigo: as duas cores estão
                # presentes em quantidade relevante no mesmo blob. Reexamina
                # numa janela um pouco maior (pra caber os dois robôs) e tenta
                # achar, dentro dela, o maior blob de CADA cor separadamente.
                # ------------------------------------------------------------
                margin = int(0.5 * winSize)
                bx1, by1 = max(0, x1 - margin), max(0, y1 - margin)
                bx2, by2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)
                hsv_big = imgHSV[by1:by2, bx1:bx2]

                for lower, upper, team_list, is_enemy in (
                    (self.ally_lower_bound, self.ally_upper_bound, ally_candidates, False),
                    (self.enemy_lower_bound, self.enemy_upper_bound, enemy_candidates, True),
                ):
                    m = self.MaskInRange(hsv_big, lower, upper)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    c = max(cnts, key=cv2.contourArea, default=None)
                    if c is None:
                        continue
                    (lx, ly), lr = cv2.minEnclosingCircle(c)
                    min_rc = (0.75 if is_enemy else 0.5) * mainColorRadius
                    if lr < min_rc:
                        continue

                    gx, gy = lx + bx1, ly + by1  # centro da cor deste robô, em coords globais

                    # Direção aproximada: do centro do blob fundido (xi,yi) até
                    # o centro da cor deste robô - mesma fórmula do caso normal;
                    # sem contorno individual próprio, é a melhor referência
                    # disponível durante a colisão.
                    direction = np.array([xi, -yi]) - np.array([gx, -gy])
                    modDir = np.linalg.norm(direction)
                    if modDir > 1e-6:
                        direction = direction / modDir

                    gxcm, gycm = self.GetPointVirtual(self.TransformPoint(np.array([gx, gy])))

                    wx1, wy1 = max(0, int(gx - half_win)), max(0, int(gy - half_win))
                    wx2, wy2 = min(img.shape[1], int(gx + half_win)), min(img.shape[0], int(gy + half_win))
                    sub_window = img[wy1:wy2, wx1:wx2]
                    if sub_window.size == 0:
                        continue

                    team_list.append({
                        "xi": gx, "yi": gy, "ri": playerRadius,
                        "x_m": gx, "y_m": gy,
                        "xcm": gxcm, "ycm": gycm, "rcm": 5.30,
                        "direction": direction,
                        "windowActual": sub_window,
                        "contour": currentPlayer,
                    })
                continue

            # --------------------------------------------------------------
            # Caso normal (sem mistura) - idêntico ao comportamento original.
            # --------------------------------------------------------------
            if ally_ratio <= 0.4 and enemy_ratio <= 0.4:
                continue  # time indefinido, mesmo comportamento de antes

            team_is_enemy = enemy_ratio > 0.4
            curr_mask = mask_enemy if team_is_enemy else mask_ally

            contour = max(cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                        key=cv2.contourArea, default=None)
            if contour is None:
                continue
            (x_m, y_m), rc = cv2.minEnclosingCircle(contour)
            x_m += x1
            y_m += y1

            min_rc = (0.75 if team_is_enemy else 0.5) * mainColorRadius
            if rc < min_rc:
                continue

            direction = np.array([xi, -yi]) - np.array([x_m, -y_m])
            modDir = np.linalg.norm(direction)
            if modDir > 1e-6:
                direction = direction / modDir

            xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([xi, yi])))

            cand = {
                "xi": xi, "yi": yi, "ri": ri,
                "x_m": x_m, "y_m": y_m,
                "xcm": xcm, "ycm": ycm, "rcm": 5.30,
                "direction": direction,
                "windowActual": windowActual,
                "contour": currentPlayer,
            }

            if team_is_enemy:
                enemy_candidates.append(cand)
            else:
                ally_candidates.append(cand)

        # --------------------------------------------------------------------
        # INIMIGOS - pareamento por vizinho mais próximo + gating (FIX #4 e #5)
        # --------------------------------------------------------------------
        if enemy_candidates:
            pairs = []
            n_slots = min(3, len(self.enemyTeam))
            for ci, cand in enumerate(enemy_candidates):
                for slot in range(n_slots):
                    last = self._enemy_last_pos.get(slot)
                    dist = 0.0 if last is None else float(np.hypot(cand["xcm"] - last[0], cand["ycm"] - last[1]))
                    pairs.append((dist, ci, slot))
            pairs.sort(key=lambda p: p[0])

            used_candidates, used_slots = set(), set()
            assignment = {}  # slot -> índice do candidato
            for dist, ci, slot in pairs:
                if ci in used_candidates or slot in used_slots:
                    continue
                last = self._enemy_last_pos.get(slot)
                if last is not None and dist > MAX_JUMP_CM:
                    continue
                assignment[slot] = ci
                used_candidates.add(ci)
                used_slots.add(slot)

            for slot, ci in assignment.items():
                cand = enemy_candidates[ci]
                bot = self.enemyTeam[slot]

                Color_p_pt, Color_s_pt, _ = self.GetCentersColors(cand["xi"], cand["yi"], cand["x_m"], cand["y_m"])
                Color_p = self.GetHsvMean(imgHSV, Color_p_pt[0], Color_p_pt[1])
                Color_s = self.GetHsvMean(imgHSV, Color_s_pt[0], Color_s_pt[1])

                self.colorTree.add_robot(ID_Team.TEAM_ENEMY, slot, self.enemyColor, Color_p, Color_s)

                if not bot.kalman_initialized:
                    bot.setPosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
                else:
                    bot.updatePosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)

                bot.updtPositionImg(cand["xi"], cand["yi"], cand["ri"])
                bot.setStatus(True)
                bot.setRadius(cand["rcm"])
                bot.setColor(colorT=self.enemyColor, colorP=Color_p, colorS=Color_s)
                self.DrawPlayerVirtual(bot)

                self._enemy_last_pos[slot] = (cand["xcm"], cand["ycm"])
                self.enemiesCount += 1

                if debug:
                    self.DrawPlayerCircle(self.frameResult, bot)
                    cx, cy = int(Color_p_pt[0]), int(Color_p_pt[1])
                    bgr_p = self.Hsv2Bgr(Color_p)
                    cv2.circle(self.frameResult, (cx, cy), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx, cy), 3, bgr_p, -1)
                    cx2, cy2 = int(Color_s_pt[0]), int(Color_s_pt[1])
                    bgr_s = self.Hsv2Bgr(Color_s)
                    cv2.circle(self.frameResult, (cx2, cy2), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx2, cy2), 3, bgr_s, -1)

        # --------------------------------------------------------------------
        # ALIADOS - identidade por cor secundária conhecida + gating (FIX #5)
        # --------------------------------------------------------------------
        for cand in ally_candidates:
            if self.alliesCount >= 3:
                break

            ally_checks = [
                (not AgoalFlag, self.goalAllyColor1, self.goalAllyColor2, ID_Robots.ROBOT_ALLY_GOAL, "Goleiro"),
                (not Aatk1Flag, self.atk1AllyColor1, self.atk1AllyColor2, ID_Robots.ROBOT_ALLY_1, "Atacante 1"),
                (not Aatk2Flag, self.atk2AllyColor1, self.atk2AllyColor2, ID_Robots.ROBOT_ALLY_2, "Atacante 2"),
            ]

            assigned = False
            for flag, c1, c2, bot_id, name in ally_checks:
                if not (flag and self.DetectAllyRobot(cand["windowActual"], c1, c2)):
                    continue

                last = self._ally_last_pos.get(bot_id)
                if last is not None:
                    dist = float(np.hypot(cand["xcm"] - last[0], cand["ycm"] - last[1]))
                    if dist > MAX_JUMP_CM:
                        # Provável contaminação de cor por um robô vizinho -
                        # ignora esse match e tenta o próximo ally_check.
                        continue

                bot = self.allyTeam[bot_id]

                if not bot.kalman_initialized:
                    bot.setPosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
                else:
                    bot.updatePosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)

                bot.updtPositionImg(cand["xi"], cand["yi"], cand["ri"])
                bot.setStatus(True)
                bot.setRadius(cand["rcm"])
                bot.setColor(colorT=self.allyColor, colorP=c1, colorS=c2)
                if debug:
                    self.DrawPlayerCircle(self.frameResult, bot)
                self.DrawPlayerVirtual(bot)

                self._ally_last_pos[bot_id] = (cand["xcm"], cand["ycm"])

                if bot_id == ID_Robots.ROBOT_ALLY_GOAL:
                    AgoalFlag = True
                elif bot_id == ID_Robots.ROBOT_ALLY_1:
                    Aatk1Flag = True
                else:
                    Aatk2Flag = True

                if debug:
                    cv2.circle(self.frameResult, (int(cand["x_m"]), int(cand["y_m"])), 4, (255, 128, 255), -1)

                assigned = True
                break

            if not assigned:
                continue

            self.alliesCount = min(self.alliesCount + 1, 3)
            if debug:
                cv2.drawContours(binaryAllies, [cand["contour"]], -1, 255, -1)
                cv2.drawContours(binaryAllTeam, [cand["contour"]], -1, 255, -1)

        if debug:
            self.binaryAllies = binaryAllies
            self.binaryAllTeam = binaryAllTeam
        self._countProcess += 1

