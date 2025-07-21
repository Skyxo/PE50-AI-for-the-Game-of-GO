# -*- coding: utf-8 -*-

import numpy as np
from const import *
import hashlib
from collections import deque

class Plateau:
    
    def __init__(self, size, plateau=None, score_capture=None, score_territoire=None, score_komi=None):
        self.__size = size

        if plateau is not None:
            self.__plateau = plateau
            self.__score_capture = score_capture
            self.__score_territoire = score_territoire
            self.__score_komi = score_komi
        else:
            self.__plateau = np.zeros((self.__size,self.__size),dtype = int)
            self.__score_capture = [0,0]
            self.__score_territoire = [0,0]
            
            if size == 9:
                self.__score_komi = [0,6.5]
            elif size == 13:
                self.__score_komi = [0,7.5]
            elif size == 19:
                self.__score_komi = [0,7.5] 

    def __str__(self):
        return f"Plateau de {self.__size}x{self.__size}."

    def get_size(self):
        return self.__size

    def clone(self):
        """
        Retourne une copie indépendante du plateau et de ses scores.
        La copie du plateau utilise ndarray.copy(), rapide et efficace.
        Les listes de scores sont dupliquées pour éviter la mutation partagée.
        """
        # Copie du tableau interne
        plateau_copy = self.__plateau.copy()
        # Copie des scores (listes)
        score_capture_copy = list(self.__score_capture)
        score_territoire_copy = list(self.__score_territoire)
        score_komi_copy = list(self.__score_komi)
        # Création du nouvel objet avec les copies
        return Plateau(
            self.__size,
            plateau=plateau_copy,
            score_capture=score_capture_copy,
            score_territoire=score_territoire_copy,
            score_komi=score_komi_copy
        )
    
    
    def hash(self):
        """Version binaire plus rapide, attend un np.array d'entiers définis par les constantes EMPTY, WHITE, BLACK"""
        plateau_np = np.array(self.__plateau, dtype=np.int8)

        # Calcul du hash à partir du tableau numpy
        return hashlib.sha256(plateau_np.tobytes()).hexdigest()
    
    
    def is_suicide(self, coup, joueur):
        """True si jouer `move` en `turn` serait un suicide."""
        # clone pour tester
        test = self.clone()
        test.add_pion(coup, joueur)
        _, libertes = test.find_chaine(coup)
        return len(libertes) == 0
    

    def is_ko(self, coup, hash_precedent, joueur):
        """True si jouer `move` crée un ko (retour à previous_hash)."""
        test = self.clone()
        test.add_pion(coup, joueur)   # ou passer turn en argument
        new_hash = test.hash()
        return new_hash == hash_precedent
        

    def get_plateau(self):
        return self.__plateau

    def get_score_territoire(self):
        return self.__score_territoire
    
    def get_score_komi(self):
        return self.__score_komi
    
    def get_score_capture(self):
        return self.__score_capture

    def get_scores(self, joueur):
        i = 0 if joueur==BLACK else 1
        return self.__score_capture[i],self.__score_territoire[i], self.__score_komi[i]

    def get_score_total(self, joueur):
        i = 0 if joueur == BLACK else 1
        # Règles chinoises : territoire + pierres sur le plateau + komi
        pierres_sur_plateau = np.sum(self.__plateau == joueur)  # Compte les pierres du joueur
        return self.__score_territoire[i] + self.__score_komi[i] + pierres_sur_plateau
    
    def get_score_total_couple(self):
        return self.get_score_total(BLACK), self.get_score_total(WHITE)

    def actualise_points_territoires(self):
        #Points des territoires
        territoire_noir, territoire_blanc = self.find_territoires()

        self.set_score_territoire(BLACK, sum([len(territoire) for territoire in territoire_noir]))
        self.set_score_territoire(WHITE, sum([len(territoire) for territoire in territoire_blanc]))
    
    def add_score_capture(self, joueur, n):
        if joueur == WHITE:
            self.__score_capture = [self.__score_capture[0],self.__score_capture[1]+n]
        else:
            self.__score_capture = [self.__score_capture[0]+n,self.__score_capture[1]]

    def set_score_territoire(self, joueur, n):
        if joueur == WHITE:
            self.__score_territoire = [self.__score_territoire[0],n]
        else:
            self.__score_territoire = [n,self.__score_territoire[1]]
    
    def supprime_chaine(self, chaine):
        for coord in chaine:
            self.__plateau[coord[0],coord[1]] = EMPTY

    def add_pion(self,coup,couleur):
        if self.__plateau[coup]!=EMPTY:
            print("déjà pion ici")
        else:
            self.__plateau[coup] = couleur
      
    def reset(self):
        self.__plateau = np.zeros((self.__size,self.__size), dtype=int) 

    def find_chaine(self, case):
        couleur = self.__plateau[case]
        visited = {case}
        case_a_parcourir = [case]
        chaine = []
        libertes = []

        while case_a_parcourir:
            case_actuelle = case_a_parcourir.pop()
            chaine.append(case_actuelle)
            for coord in self.get_voisins(case_actuelle):
                if coord not in visited:
                    visited.add(coord)
                    if self.__plateau[coord] == couleur:
                        case_a_parcourir.append(coord)
                    elif self.__plateau[coord] == EMPTY:
                        libertes.append(coord)

        return chaine, libertes
        
    def get_voisins(self, coord):
        x, y = coord

        d_list = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        voisins = []

        for dx, dy in d_list:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.__size and 0 <= ny < self.__size:
                voisins.append((nx, ny))

        return voisins
    

    def find_territoires(self):
        size = self.__size
        plateau = self.__plateau

        cases_vides_a_traiter = {
            (i, j)
            for i in range(size)
            for j in range(size)
            if plateau[i, j] == EMPTY
        } # toutes les cases vides

        territoire_noir, territoire_blanc = [], []

        while cases_vides_a_traiter: # tant qu'on n'a pas parcouru toutes les cases vides du plateau
            start = cases_vides_a_traiter.pop() # première case vide
            territoire = [start]
            stack = [start] # territoire à parcourir autour de la case vide
            seen_black, seen_white = False, False

            # Explorer complètement le territoire d'abord
            while stack:
                case_actuelle = stack.pop()
                for vois in self.get_voisins(case_actuelle): # on regarde les voisins de la case actuelle
                    val = plateau[vois]

                    if val == EMPTY and vois in cases_vides_a_traiter: # si un voisin est une case vide
                        stack.append(vois) # on le parcourra plus tard
                        territoire.append(vois) # on l'ajoute au territoire
                        cases_vides_a_traiter.remove(vois) # on le retire des cases vides à traiter
                    
                    elif val == BLACK: # si la pierre est noire
                        seen_black = True # on le notifie
                    elif val == WHITE: # si la pierre est blanche
                        seen_white = True # on le notifie

            # Une fois le territoire complètement exploré, on détermine à qui il appartient
            if seen_black and not seen_white: # s'il n'y a pas de blancs à la frontière
                territoire_noir.append(territoire) # c'est un territoire noir
            elif seen_white and not seen_black: # s'il n'y a pas de noirs à la frontière
                territoire_blanc.append(territoire) # c'est un territoire blanc

        return territoire_noir, territoire_blanc


    def compter_pierres(self):
        nb_noires = 0
        nb_blanches = 0
        for x in range(self.__size):
            for y in range(self.__size):
                if self.__plateau[x,y] == WHITE :
                    nb_noires += 1
                if self.__plateau[x,y] == BLACK:
                    nb_blanches += 1
        return (nb_noires,nb_blanches)

    def group_in_enemy_territory(self, start, player):
        """
        Renvoie True si le groupe de `player` contenant la case `start`
        est entièrement situé dans le territoire de l'adversaire.
        """
        opponent = WHITE if player == BLACK else BLACK
        
        # 1) Collecte du groupe + de ses libertés
        groupe, libertes = self.find_chaine(start)
        
        # Si aucun groupe trouvé ou pas une pierre du joueur spécifié
        if not groupe or self.__plateau[start] != player:
            return False
        
        # 2) Si le groupe a deux yeux, il n'est pas en territoire adverse
        if self._a_deux_yeux(groupe, player):
            return False
        
        # 3) Vérifier si le groupe est entièrement entouré par l'adversaire
        # en utilisant la fonction existante
        return self._est_entoure_par_adversaire(groupe, player, libertes)

        """
        Renvoie True si le groupe de `player` contenant la case `start`
        est entièrement situé dans le territoire de l'adversaire.
        """
        opponent = WHITE if player == BLACK else BLACK
        
        # 1) Collecte du groupe + de ses libertés
        groupe, libertes = self.find_chaine(start)
        groupe_set = set(groupe)  # Pour recherche efficace
        
        # Si aucun groupe trouvé ou pas une pierre du joueur spécifié
        if not groupe or self.__plateau[start] != player:
            return False
        
        # 2) Pour chaque liberté, étudier sa zone vide
        seen_empty = np.zeros((self.__size, self.__size), dtype=bool)
        
        for liberte in libertes:
            if not seen_empty[liberte]:
                # Passer le groupe comme paramètre
                edge, touches_player, touches_opponent = self._empty_region_info(
                    liberte, player, opponent, seen_empty, groupe_set
                )
                
                if edge or touches_player or not touches_opponent:
                    return False
        
        return True

    def _empty_region_info(self, start, player, opponent, seen_empty, groupe_set):
        """
        Analyse une région vide connectée.
        """
        touches_edge = touches_player = touches_opponent = False
        region = deque([start])
        seen_empty[start] = True
        
        while region:
            current = region.popleft()
            x, y = current
            
            # Vérifier si on est sur le bord
            if x == 0 or y == 0 or x == self.__size - 1 or y == self.__size - 1:
                touches_edge = True
            
            # Explorer le voisinage
            for voisin in self.get_voisins(current):
                val = self.__plateau[voisin]
                
                if val == EMPTY and not seen_empty[voisin]:
                    seen_empty[voisin] = True
                    region.append(voisin)
                elif val == player and voisin not in groupe_set:  # ← CORRECTION ICI
                    touches_player = True
                elif val == opponent:
                    touches_opponent = True
        
        return touches_edge, touches_player, touches_opponent


    def detecter_pierres_encerclees(self, position_depart):
        """
        Détecte si une chaîne fermée de pierres contient des pierres adverses et les compte.
        
        Args:
            position_depart: Position d'une pierre de la chaîne encerclante
            
        Returns:
            dict: {
                'est_fermee': bool,             # True si la chaîne forme un cercle fermé
                'chaine': list,                # Les coordonnées des pierres de la chaîne
                'pierres_encerclees': list,     # Les coordonnées des pierres adverses encerclées
                'nombre': int                   # Le nombre de pierres adverses encerclées
            }
        """
        # Vérifier la couleur de la pierre de départ
        couleur = self.__plateau[position_depart]
        if couleur not in [BLACK, WHITE]:
            return {'est_fermee': False, 'chaine': [], 'pierres_encerclees': [], 'nombre': 0}
        
        # Obtenir la chaîne complète à partir de la position de départ
        chaine, _ = self.find_chaine(position_depart)
        
        # Trouver les espaces entourés par cette chaîne
        espaces_interieurs = self._trouver_espaces_interieurs_chaine(chaine)
        
        # Si aucun espace intérieur trouvé, la chaîne n'est pas fermée
        if not espaces_interieurs:
            return {'est_fermee': False, 'chaine': chaine, 'pierres_encerclees': [], 'nombre': 0}
        
        # Compter les pierres adverses dans les espaces intérieurs
        couleur_adverse = WHITE if couleur == BLACK else BLACK
        pierres_encerclees = []
        
        for espace in espaces_interieurs:
            for pos in espace:
                if self.__plateau[pos] == couleur_adverse:
                    pierres_encerclees.append(pos)
        
        return {
            'est_fermee': True,
            'chaine': chaine,
            'pierres_encerclees': pierres_encerclees,
            'nombre': len(pierres_encerclees)
        }

    def _trouver_espaces_interieurs_chaine(self, chaine):
        """
        Trouve les espaces intérieurs délimités par une chaîne de pierres.
        
        Args:
            chaine: Liste de coordonnées formant la chaîne
        
        Returns:
            list: Liste des espaces intérieurs (chaque espace est une liste de coordonnées)
        """
        # Créer un tableau temporaire pour marquer la chaîne
        marquage = np.zeros((self.__size, self.__size), dtype=int)
        for pos in chaine:
            marquage[pos] = 1  # 1 = pierre de la chaîne
        
        # Marquer l'espace extérieur en commençant par les bords
        visite = np.zeros_like(marquage, dtype=bool)
        queue = []
        
        # Ajouter tous les points du bord comme points de départ
        for i in range(self.__size):
            queue.extend([(i, 0), (i, self.__size - 1)])
        for j in range(1, self.__size - 1):
            queue.extend([(0, j), (self.__size - 1, j)])
        
        # Marquer les points de départ comme visités
        for pos in queue:
            if marquage[pos] == 0:  # Ne marquer que si ce n'est pas une pierre de la chaîne
                visite[pos] = True
        
        # BFS pour marquer l'espace extérieur accessible
        while queue:
            x, y = queue.pop(0)
            for nx, ny in self.get_voisins((x, y)):
                if not visite[nx, ny] and marquage[nx, ny] == 0:
                    visite[nx, ny] = True
                    queue.append((nx, ny))
        
        # Les espaces non visités et non occupés sont des espaces intérieurs
        espaces_interieurs = []
        for i in range(self.__size):
            for j in range(self.__size):
                if marquage[i, j] == 0 and not visite[i, j]:
                    # Nouvel espace intérieur trouvé
                    espace = []
                    queue = [(i, j)]
                    visite[i, j] = True
                    
                    while queue:
                        x, y = queue.pop(0)
                        espace.append((x, y))
                        for nx, ny in self.get_voisins((x, y)):
                            if not visite[nx, ny] and marquage[nx, ny] == 0:
                                visite[nx, ny] = True
                                queue.append((nx, ny))
                    
                    espaces_interieurs.append(espace)
        
        return espaces_interieurs

    def get_groupes_morts(self):
        """
        Détecte et retire les groupes morts du plateau.
        Un groupe est considéré mort s'il est entouré par des pierres adverses et ne peut pas former deux yeux.
        
        Returns:
            list: Liste des groupes morts (chaque groupe est une liste de coordonnées)
        """
        groupes_morts = []
        groupes_vivants = set()  # Pour mémoriser les coordonnées des groupes déjà analysés comme vivants
        
        # Parcourir tout le plateau pour trouver les groupes
        for i in range(self.__size):
            for j in range(self.__size):
                case = (i, j)
                if case in groupes_vivants or self.__plateau[case] == EMPTY:
                    continue
                    
                # Vérifier si cette case fait déjà partie d'un groupe identifié comme mort
                if any(case in groupe for groupe in groupes_morts):
                    continue
                    
                couleur = self.__plateau[case]
                groupe, libertes = self.find_chaine(case)
                
                # Un groupe sans libertés ne devrait pas exister (serait déjà capturé)
                if not libertes:
                    continue
                    
                # Vérifier si le groupe est vivant (a deux yeux)
                if self._a_deux_yeux(groupe, couleur):
                    # Ce groupe est vivant, marquer toutes ses pierres
                    groupes_vivants.update(groupe)
                else:
                    # Vérifier si le groupe est entouré par l'adversaire
                    if self._est_entoure_par_adversaire(groupe, couleur, libertes):
                        groupes_morts.append(groupe)
                    else:
                        # Si non entouré, il n'est pas encore mort
                        groupes_vivants.update(groupe)
                    
        # Retirer les groupes morts et ajuster les scores
        for groupe in groupes_morts:
            couleur = self.__plateau[groupe[0]]
            adversaire = WHITE if couleur == BLACK else BLACK
            
            # Supprimer le groupe
            self.supprime_chaine(groupe)
            
            # Ajuster le score (capturer)
            self.add_score_capture(adversaire, len(groupe))
                    
        return groupes_morts

    def _est_entoure_par_adversaire(self, groupe, couleur, libertes):
        """
        Vérifie si un groupe est entouré par l'adversaire de manière stratégique.
        Un groupe est considéré encerclé si toutes ses libertés sont dans un territoire 
        contrôlé par l'adversaire, même avec des espaces.
        """
        adversaire = WHITE if couleur == BLACK else BLACK
        
        # Si le groupe a déjà deux yeux, il n'est pas encerclé
        if self._a_deux_yeux(groupe, couleur):
            return False
        
        # Pour chaque liberté, vérifier si elle est dans la zone d'influence adverse
        for liberte in libertes:
            # Effectuer une recherche en largeur à partir de cette liberté
            # pour voir si on peut atteindre une zone non contrôlée par l'adversaire
            queue = [liberte]
            visited = {liberte}
            peut_s_echapper = False
            max_profondeur = min(7, self.__size)  # Limiter la recherche pour l'efficacité
            
            # Pour chaque liberté, explorer jusqu'à max_profondeur
            profondeur = 0
            niveau_actuel = 1  # Nombre d'éléments au niveau actuel
            niveau_suivant = 0  # Nombre d'éléments au niveau suivant
            
            while queue and profondeur < max_profondeur:
                current = queue.pop(0)
                niveau_actuel -= 1
                
                # Vérifier si on trouve une zone qui n'est pas sous contrôle adverse
                if not self._case_sous_controle_adverse(current, adversaire):
                    peut_s_echapper = True
                    break
                    
                # Explorer les voisins
                for voisin in self.get_voisins(current):
                    # N'explorer que les cases vides non visitées
                    if self.__plateau[voisin] == EMPTY and voisin not in visited:
                        visited.add(voisin)
                        queue.append(voisin)
                        niveau_suivant += 1
                
                # Passage au niveau suivant dans le BFS
                if niveau_actuel == 0:
                    niveau_actuel = niveau_suivant
                    niveau_suivant = 0
                    profondeur += 1
                    
            # Si une liberté peut s'échapper, le groupe n'est pas encerclé
            if peut_s_echapper:
                return False
                
        return True  # Toutes les libertés sont effectivement encerclées

    def _case_sous_controle_adverse(self, case, adversaire):
        """
        Détermine si une case est sous le contrôle de l'adversaire.
        Une case est sous contrôle adverse si:
        1. Elle est entourée majoritairement par des pierres adverses
        2. Elle n'est pas à côté de nos propres pierres (sauf le groupe évalué)
        """
        notre_couleur = BLACK if adversaire == WHITE else WHITE
        pierres_adverses = 0
        pierres_alliees = 0
        
        for voisin in self.get_voisins(case):
            if self.__plateau[voisin] == adversaire:
                pierres_adverses += 1
            elif self.__plateau[voisin] == notre_couleur:
                pierres_alliees += 1
        
        # Si entourée principalement par l'adversaire, considérée sous contrôle adverse
        return pierres_adverses > 0 and pierres_adverses > pierres_alliees
    
    def _a_deux_yeux(self, groupe, couleur):
        """
        Détermine si un groupe a au moins deux yeux séparés.
        Un œil est un espace vide entouré uniquement par des pierres du groupe.
        """
        # Trouver toutes les libertés du groupe
        libertes = set()
        for case in groupe:
            for voisin in self.get_voisins(case):
                if self.__plateau[voisin] == EMPTY:
                    libertes.add(voisin)
        
        # Si moins de 2 libertés, impossible d'avoir 2 yeux
        if len(libertes) < 2:
            return False
        
        # Regrouper les libertés connectées (chaque groupe = un œil potentiel)
        yeux = []
        libertes_restantes = libertes.copy()
        
        while libertes_restantes:
            liberte = libertes_restantes.pop()
            oeil = {liberte}
            a_explorer = [liberte]
            
            # Explorer les libertés connectées
            while a_explorer:
                case = a_explorer.pop()
                for voisin in self.get_voisins(case):
                    if self.__plateau[voisin] == EMPTY and voisin in libertes_restantes:
                        libertes_restantes.remove(voisin)
                        oeil.add(voisin)
                        a_explorer.append(voisin)
            
            # Vérifier si cet œil est un vrai œil (entouré uniquement par le groupe)
            est_vrai_oeil = True
            for liberte in oeil:
                for voisin in self.get_voisins(liberte):
                    # Un vrai œil est entouré uniquement par des pierres de notre couleur
                    if self.__plateau[voisin] != couleur and voisin not in oeil:
                        est_vrai_oeil = False
                        break
                if not est_vrai_oeil:
                    break
                    
            if est_vrai_oeil:
                yeux.append(oeil)
        
        # S'il y a au moins deux vrais yeux, le groupe est vivant
        return len(yeux) >= 2
    
    def afficher_plateau(self):
        """
        Affiche le plateau dans la console avec des symboles pour les cases.
        """
        symboles = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        for row in self.__plateau:
            print(' '.join(symboles[cell] for cell in row))
    
"""
    def retire_groupes_morts(self):
        #Retire les groupes morts du plateau.
        groupes_morts_noir, groupes_morts_blanc = self.detect_groupes_morts()

        # Retirer les pierres mortes du plateau
        for groupe in groupes_morts_noir:
            for (x, y) in groupe:
                self.__plateau[x, y] = 0  # Remplacer par une intersection vide

        for groupe in groupes_morts_blanc:
            for (x, y) in groupe:
                self.__plateau[x, y] = 0

        self.add_score_capture(-1, len(sum(groupes_morts_noir, [])))
        self.add_score_capture(1, len(sum(groupes_morts_blanc, [])))
"""
