from copy import deepcopy
import numpy as np
from plateau import Plateau
import MCTS_GO as mcts
from const import *
import torch
import os

class BaseMoteurDeJeu:
    """Superclasse contenant les fonctionnalités de base du moteur de jeu"""

    def __init__(self, plateau, turn=BLACK, compteur_pass=0, hash_plateaux=None):
        self.__plateau = plateau
        self.__turn = turn
        self.__compteur_pass = compteur_pass
        self.__plateau_provisoire = plateau
        self.__hash_plateaux = hash_plateaux if hash_plateaux else []
        
    def set_turn(self, turn):
        self.__turn = turn

    def to_tensor(self):
        """
        Convertit l'état du jeu en un tenseur (channels x 19 x 19)
        - Channel 0 : pierres noires
        - Channel 1 : pierres blanches
        - Channel 2 : 1 si c'est au tour du joueur noir, 0 sinon
        """
        plateau = self.__plateau.get_plateau()  # Correction ici

        black = (plateau == 1).astype(np.float32)
        white = (plateau == 2).astype(np.float32)
        turn_plane = np.ones_like(black, dtype=np.float32) if self.__turn == 1 else np.zeros_like(black)

        stacked = np.stack([black, white, turn_plane], axis=0)  # shape (3, 19, 19)
        return torch.tensor(stacked)


    def clone(self):
        """
        Retourne une copie indépendante du moteur, avec :
         - un clone du plateau,
         - le même turn,
         - le même compteur de passes,
         - une copie de la liste des hashes pour le ko.
        """
        # 1) clone le plateau
        plat_copy = self.__plateau.clone()
        # 2) copie superficielle des autres attributs
        turn_copy = self.__turn
        pass_copy = self.__compteur_pass
        hashes_copy = list(self.__hash_plateaux)
        # 3) construit un nouveau moteur de simulation
        return simulation_MoteurDeJeu(plat_copy, turn_copy, pass_copy, hashes_copy)


    def get_plateau_provisoire(self):
        return self.__plateau_provisoire
    

    def get_hash_plateaux(self):
        return self.__hash_plateaux
    

    def get_plateau(self):
        return self.__plateau
    

    def get_turn(self):
        return self.__turn


    def get_pass(self):
        return self.__compteur_pass


    def add_pass(self):
        self.__compteur_pass += 1


    def get_winner_and_score(self):
        """
        Détermine le gagnant et l'écart de points selon les règles chinoises.
        Retourne: (couleur du vainqueur, écart de points, score noir, score blanc)
        """
        self.get_plateau().actualise_points_territoires()
        score_noir, score_blanc = self.get_plateau().get_score_total_couple()
        
        if score_noir > score_blanc:
            return BLACK, score_noir - score_blanc, score_noir, score_blanc
        else:
            return WHITE, score_blanc - score_noir, score_noir, score_blanc


    def update_plateau(self):
        self.__plateau = self.__plateau_provisoire
        self.__compteur_pass = 0
        # historique des ko
        if len(self.__hash_plateaux) > 1:
            self.__hash_plateaux.pop(0)
        self.__hash_plateaux.append(self.__plateau.hash())


    def changer_joueur(self):
        """Change le tour du joueur."""
        self.__turn = next_player(self.__turn)


    def nouveau_plateau(self, case):
        """Simule le plateau après un coup."""
        plateau = Plateau(self.__plateau.get_size(), 
                          self.__plateau.get_plateau().copy(),
                          self.__plateau.get_score_capture(),
                          self.__plateau.get_score_territoire(),
                          self.__plateau.get_score_komi())

        if plateau.get_plateau()[case] != EMPTY:
            return
        
        plateau.add_pion(case, self.__turn)
        goban = plateau.get_plateau()

        for case_voisine in plateau.get_voisins(case):

            if goban[case_voisine] != next_player(self.__turn):
                continue
            
            chaine_voisin, liberte_voisin = plateau.find_chaine(case_voisine)
            
            if len(liberte_voisin) == 0:
                plateau.supprime_chaine(chaine_voisin)
                plateau.add_score_capture(self.__turn, len(chaine_voisin))
                goban = plateau.get_plateau()

        self.__plateau_provisoire = plateau


    def regle_suicide(self, coup):
        # changeable en envoyant juste coup dans find chaine en modifiant find chaine en find chaine rule en donnant la couleur en paramètre
        """Vérifie si le coup respecte la règle du suicide."""
        #print("regle suicide")
        libertes = self.__plateau_provisoire.find_chaine(coup)[1]
        #print(libertes, libertes_simule)
        return len(libertes) != 0
    
    
    def regle_du_ko(self):
        """Vérifie si la règle du ko est respectée en comparant les hashs."""
        if len(self.__hash_plateaux) > 1:
            return self.__plateau_provisoire.hash() != self.__hash_plateaux[0]
        return True


    def coup_jouable(self, coup):

        if coup is None:
            return True

        if self.__plateau.get_plateau()[coup] == EMPTY:
            self.nouveau_plateau(coup)
            suicide = self.regle_suicide(coup)
            ko = self.regle_du_ko()
            return ko and suicide
        
        return False
    
    
    def liste_coups_jouable(self):
        if self.end_game():
            return []

        arr  = self.__plateau.get_plateau()
        N    = self.__plateau.get_size()
        coups = [
            (i, j)
            for i in range(N)
            for j in range(N)
            if arr[i, j] == EMPTY and self.coup_jouable((i, j))
        ]
        # On inclut toujours une option pass pour qu'il puisse arriver
        # deux passes de suite et mettre fin à la partie
        return coups + [None]


    def end_game(self):
        """Vérifie si la partie est terminée."""
        return self.__compteur_pass == 2
    

    def compter_points(self):
        """Calcule et affiche les points."""
        self.__plateau.actualise_points_territoires()
        self.affiche_score()


    def affiche_score(self):
        """Affiche les scores des joueurs selon les règles chinoises."""
        pl = self.get_plateau()
        if pl is None:
            print("Erreur : plateau non initialisé.")
            return
        score_n, score_b = pl.get_score_total_couple()
        territoire = pl.get_score_territoire()
        komi = pl.get_score_komi()
        if territoire is None or komi is None:
            print("Erreur : territoire ou komi non initialisé.")
            return
        territoire_n = territoire[0]
        territoire_b = territoire[1]
        komi_n, komi_b = komi[0], komi[1]
        pierres_n = np.sum(pl.get_plateau() == BLACK)
        pierres_b = np.sum(pl.get_plateau() == WHITE)

        print(f'Noir ({score_n} pts) :\n territoire : {territoire_n} pts\n pierres : {pierres_n} pts\n komi : {komi_n} pts\n')
        print(f'Blanc ({score_b} pts) :\n territoire : {territoire_b} pts\n pierres : {pierres_b} pts\n komi : {komi_b} pts\n')

        print(f'Le joueur {"noir" if score_n > score_b else "blanc"} gagne !')


# ---------------------- Classe MoteurDeJeu ----------------------------------
class MoteurDeJeu(BaseMoteurDeJeu):

    def __init__(self, gobanGraphique, ai, ai_player, size, sbx, mcts_iterations=2500):
        plateau = Plateau(size)
        turn = BLACK if not ai else next_player(ai_player)
        super().__init__(plateau, turn)
        self.__gobanGraphique = gobanGraphique
        self.__ai = ai
        self.__ai_player = ai_player
        self.__sbx = sbx
        self.__mcts_iterations = mcts_iterations
        if ai:
            # Utiliser les réseaux de neurones policy et value
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def set_mcts_iterations(self, iterations):
        """Met à jour le nombre d'itérations MCTS et réinitialise le mode"""
        self.__mcts_iterations = iterations
        if self.__ai:
            # Réinitialiser avec les mêmes paramètres mais nouveau nombre d'itérations
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def jouer_premier_coup(self):
        coup, _ = mcts.MCTS(self, self.__ai_player)
        self.jouer_coup(coup)

    def jouer_automatiquement(self):
        """Joue automatiquement les coups pour l'IA"""
        while not self.end_game():
            coup, _ = mcts.MCTS(self, self.get_turn())
            self.jouer_coup(coup)

    def jouer_coup(self, coup):
        """Joue un coup et met à jour le plateau."""

        if self.end_game():
            print("La partie est déjà terminée.")
            return

        joueur = 'noir' if self.get_turn() == BLACK else 'blanc'

        if not self.coup_jouable(coup):
            print("Coup impossible :", coup, "| Tour du joueur :", joueur)
            return        

        if coup is None:
            print(f"Le joueur {joueur} passe son tour.")
            self.add_pass()
        else:
            self.update_plateau()
            self.__gobanGraphique.update_affichage()

        # On change de joueur si on n'est pas en sandbox
        if coup is None or not self.__sbx:
            self.changer_joueur()

            # Si l'IA doit jouer, on appelle récursivement
            if self.__ai and self.get_turn() == self.__ai_player and not self.end_game():
                prochain_coup, _ = mcts.MCTS(self, self.__ai_player)
                self.jouer_coup(prochain_coup)

        # Fin de partie
        if self.end_game():
            self.get_plateau().actualise_points_territoires()
            self.__gobanGraphique.dessiner_territoires()
            print("\n------------- Partie terminée -------------")
            self.get_plateau().get_groupes_morts()
            self.affiche_score()

# ---------------------- Classe simulation_MoteurDeJeu ------------------------
class simulation_MoteurDeJeu(BaseMoteurDeJeu):

    def __init__(self, plateau, turn=BLACK, compteur_pass=0, hash_plateaux=None):
        super().__init__(plateau, turn, compteur_pass, hash_plateaux)


    def jouer_coup(self, coup):
        """Simule un coup."""
        if self.end_game():
            print("La partie est terminée. (simulation)")
        elif self.coup_jouable(coup):
            if coup:
                self.update_plateau()
            else:
                self.add_pass()
            self.changer_joueur()
        else:
            print("Coup impossible (simulation)", coup, self.get_turn())

# ---------------------- Classe MoteurDeJeuAIvsAI ----------------------------------
class MoteurDeJeuAIvsAI(BaseMoteurDeJeu):

    def __init__(self, gobanGraphique, ai, ai_player, size, sbx, mcts_iterations=2500, config=None):
        plateau = Plateau(size)
        turn = BLACK if not ai else next_player(ai_player)
        super().__init__(plateau, turn)
        self.__gobanGraphique = gobanGraphique
        self.__ai = ai
        self.__ai_player = ai_player
        self.__sbx = sbx
        self.__mcts_iterations = mcts_iterations
        self.__config = config  # Configuration pour IA vs IA
        if ai:
            # Utiliser les réseaux de neurones policy et value
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def set_mcts_iterations(self, iterations):
        """Met à jour le nombre d'itérations MCTS et réinitialise le mode"""
        self.__mcts_iterations = iterations
        if self.__ai:
            # Réinitialiser avec les mêmes paramètres mais nouveau nombre d'itérations
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def jouer_premier_coup(self):
        """Joue le premier coup pour l'IA"""
        if self.__ai and self.__ai_player == BLACK:
            coup, _ = mcts.MCTS(self, self.__ai_player)
            self.jouer_coup(coup)

    def jouer_automatiquement(self):
        """Joue automatiquement les coups pour l'IA vs IA"""
        while not self.end_game():
            coup, _ = mcts.MCTS(self, self.get_turn())
            self.jouer_coup(coup)

    def jouer_coup(self, coup):
        """Joue un coup et met à jour le plateau."""
        if self.end_game():
            print("La partie est terminée.")
            return

        if self.coup_jouable(coup):
            if coup is None:
                print(f"Le joueur {'noir' if self.get_turn() == BLACK else 'blanc'} passe son tour.")
                self.add_pass()
            else:
                self.update_plateau()
                if self.__gobanGraphique:
                    self.__gobanGraphique.update_affichage()

            self.changer_joueur()
        else:
            print("Coup impossible", coup, self.get_turn())



from CNN.resume_training import PolicyNet

class MoteurDeJeuCNN(BaseMoteurDeJeu):
    def __init__(self, gobanGraphique, ai, ai_player, size, sbx, mcts_iterations=2500):
        plateau = Plateau(size)
        turn = BLACK if not ai else next_player(ai_player)
        super().__init__(plateau, turn)
        self.__gobanGraphique = gobanGraphique
        self.__ai = ai
        self.__ai_player = ai_player
        self.__sbx = sbx
        self.__mcts_iterations = mcts_iterations
        if ai:
            # Utiliser les réseaux de neurones policy et value
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def set_mcts_iterations(self, iterations):
        """Met à jour le nombre d'itérations MCTS et réinitialise le mode"""
        self.__mcts_iterations = iterations
        if self.__ai:
            # Réinitialiser avec les mêmes paramètres mais nouveau nombre d'itérations
            policy_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                     "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                                     "best_policynet_9x9.pth")
            value_path = os.path.join("jeu_de_go", "katago_relatives", "katago-opencl-linux", "networks",
                                    "training_logs_value9x9_v3_2025-06-06_18-21-00",
                                    "best_valuenet_9x9.pth")
            mcts.enable_policy_value_networks(iterations=self.__mcts_iterations, 
                                           exploration=1.1,
                                           policy_path=policy_path,
                                           value_path=value_path)

    def jouer_premier_coup(self):
        """Joue le premier coup pour l'IA"""
        if self.__ai and self.__ai_player == BLACK:
            coup, _ = mcts.MCTS(self, self.__ai_player)
            self.jouer_coup(coup)

    def jouer_automatiquement(self):
        """Joue automatiquement les coups pour l'IA"""
        while not self.end_game():
            coup, _ = mcts.MCTS(self, self.get_turn())
            self.jouer_coup(coup)

    def jouer_coup(self, coup):
        """Joue un coup et met à jour le plateau."""
        if self.end_game():
            print("La partie est terminée.")
            return

        joueur = 'noir' if self.get_turn() == BLACK else 'blanc'
        
        if not self.coup_jouable(coup):
            print("Coup impossible :", coup, "| Tour du joueur :", joueur)
            return

        if coup is None:
            print(f"Le joueur {joueur} passe son tour.")
            self.add_pass()
        else:
            self.update_plateau()
            self.__gobanGraphique.update_affichage()

        self.changer_joueur()

        # Si c'est au tour de l'IA
        if self.get_turn() == self.__ai_player and not self.end_game():
            coup_ia, _ = mcts.MCTS(self, self.__ai_player)
            self.jouer_coup(coup_ia)

        # Fin de partie
        if self.end_game():
            self.get_plateau().actualise_points_territoires()
            self.__gobanGraphique.dessiner_territoires()
            print("\n------------- Partie terminée -------------")
            self.get_plateau().get_groupes_morts()
            self.affiche_score()


"""
import re

def parse_sgf_moves(sgf_str):
    # Extrait les coups du SGF sous forme de liste [('B', 'ee'), ('W', 'dg'), ...]
    pattern = re.compile(r';([BW])\[([a-i]{0,2})\]')
    moves = pattern.findall(sgf_str)
    return moves

def sgf_coord_to_tuple(coord):
    # Convertit 'ee' en (4, 4), 'dg' en (3, 6), etc. (SGF: a=0, b=1, ...)
    if coord == '':
        return None  # Pass
    return (ord(coord[0]) - ord('a'), ord(coord[1]) - ord('a'))

def play_sgf_game(sgf_str):
    moves = parse_sgf_moves(sgf_str)
    plateau = Plateau(9)
    moteur = simulation_MoteurDeJeu(plateau)
    color_map = {'B': BLACK, 'W': WHITE}

    for color, coord in moves:
        coup = sgf_coord_to_tuple(coord)
        # On force le tour si besoin (pour être sûr d'être synchro avec le SGF)
        moteur.set_turn(color_map[color])
        moteur.jouer_coup(coup)
    
    print("Plateau final :")
    moteur.get_plateau().afficher_plateau()
    return moteur.get_plateau()

sgf = "(;FF[4]GM[1]SZ[9]PB[Current_Model]PW[Previous_Model]RE[W+18.5];B[ee];W[dg];B[cf];W[fg];B[gd];W[cg];B[cd];W[gg];B[fc];W[bf];B[be];W[he];B[hd];W[eg];B[hf];W[bg];B[hg];W[hh];B[ig];W[ff];B[ge];W[ih];B[ce];W[fe];B[fd];W[ae];B[bc];W[ad];B[ac];W[ef];B[ed];W[ii];B[bd];W[gi];B[dd];W[af];B[de];W[gf];B[ie];W[fi];B[cc];W[eb];B[df];W[fb];B[];W[fa];B[db];W[gh];B[gb];W[da];B[];W[])"
plate = play_sgf_game(sgf)

"""