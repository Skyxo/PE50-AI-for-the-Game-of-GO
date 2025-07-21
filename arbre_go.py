from noeud_go import *
from copy import deepcopy

class Arbre:

    def __init__(self,graphe:dict):
        self.__graphe = graphe

    def get_graphe(self):
        return self.__graphe
    
    def get_chemin_coups(self, noeud):
        
        res = [(noeud.get_coup(), noeud.get_joueur())]
        explore = [noeud]

        while explore:
            node = explore.pop(0)
            for k in self.__graphe.keys():
                if node in self.__graphe[k]:
                    explore.append(k)
                    res.insert(0, (k.get_coup(), k.get_joueur()))
        
        return res

    def affiche_graphe(self):
    
        for k, v in self.__graphe.items():
            if k.get_id()<26:
                print(f"Noeud {k.get_id()} (joueur {k.get_joueur()}, coup {k.get_coup()}, win/sim : {k.get_value()}/{k.get_nombre_simulation()} = {k.get_value()/k.get_nombre_simulation() if k.get_nombre_simulation() else 'inf'}) parent : {self.get_parent(k).get_id() if self.get_parent(k) else None} (joueur {self.get_parent(k).get_joueur() if self.get_parent(k) else None}); enfants : {[noeud.get_id() for noeud in v]} ; chemin coups : {self.get_chemin_coups(k)}")#; plateau : \n{k.get_etat().get_plateau().get_plateau()}\n")
                k.affiche_noeuds_rollout()
        print()
    
    def get_new_indice(self):
        lst = [noeud.get_id() for noeud in self.__graphe.keys()]
        return 0 if len(lst)==0 else max(lst)+1
    
    def add_noeud(self,parent,etat,joueur,coup):
        
        noeud = Noeud(etat,joueur,coup)
        self.__graphe[noeud] = []
        if parent!= None:
            self.__graphe[parent].append(noeud)
        return noeud  # Retourner le noeud créé
    
    def add_noeud_neuille(self,noeud,parent):
        self.__graphe[noeud] = []
        if parent!= None:
            self.__graphe[parent].append(noeud)

    def get_enfants(self, noeud):
        return self.__graphe[noeud]
    
    def get_parent(self, noeud):
        for k in self.__graphe.keys():
            if noeud in self.__graphe[k]:
                return k
        return None
    
    def get_racine(self):
        return list(self.__graphe.keys())[0]

    def remplace_racine(self, noeud):
        new_graphe = {}
        file = [noeud]

        while len(file) > 0:
            current_node = file.pop(0)
            new_graphe[current_node] = []

            for enfant in self.get_enfants(current_node):
                new_graphe[current_node].append(enfant)
                file.append(enfant)

        self.__graphe = new_graphe
