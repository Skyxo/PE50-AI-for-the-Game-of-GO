class Noeud:

    def __init__(self, etat, joueur, coup):
        self.__nombre_simulation = 0
        self.__value = 0
        self.__win = 0
        self.__loose = 0
        self.__etat = etat
        self.__joueur = joueur
        self.__coup = coup
        self.__noeuds_rollout = []
        self.__prior = 0.0  # Ajouter une valeur de prior pour le CNN
        self.__id = id(self)  # Utilisation de l'id d'objet pour résoudre le problème d'identifiant unique

    #def __str__(self):
    #    return f"Noeud avec {self.__value} valeur pour {self.__nombre_simulation} passages"

    def get_id(self):
        return self.__id
    
    def get_value(self):
        return self.__value

    def get_loose(self):
        return self.__loose

    def get_win(self):
        return self.__win

    def get_nombre_simulation(self):
        return self.__nombre_simulation
    
    def get_etat(self):
        return self.__etat
    
    def get_joueur(self):
        return self.__joueur
    
    def get_coup(self):
        return self.__coup
    
    def add_loose(self):
        self.__loose+=1
    
    def add_win(self):
        self.__win+=1

    def actualise_simulation_et_value(self, value):
        self.__nombre_simulation += 1
        self.__value += value

    def ajoute_rollout(self, node):
        self.__noeuds_rollout.append(node)

    def set_prior(self, prior):
        self.__prior = prior
        
    def get_prior(self):
        return self.__prior

    def affiche_noeuds_rollout(self):
        for noeud in self.__noeuds_rollout:
            print(f"   Noeud parent {self.__id}; Noeud n°{noeud.get_id()} val/sim : {noeud.get_value()}/{noeud.get_nombre_simulation()} ; plateau :\n{noeud.get_etat().get_plateau().get_plateau()}")