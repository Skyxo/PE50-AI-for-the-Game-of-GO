from tkinter import *
from moteursDeJeu import *
import os
from const import *
import torch

class GobanGraphique(Toplevel):

    def __init__(self, menuGraphique, ai, ai_player, size, sbx, ai_vs_ai, cnn=False, mcts_iterations=2500):
        self.__menuGraphique = menuGraphique
        self.__size = size
        self.__cell_size = 30
        self.__goban_size = self.__cell_size * (self.__size - 1) + 60
        self.__sbx = sbx
        self.__preview = False
        self.__preview_oval = None
        self.__mcts_iterations = mcts_iterations
        self.__ai = ai
        self.__ai_player = ai_player
        self.__ai_vs_ai = ai_vs_ai
        self.__cnn = cnn

        Toplevel.__init__(self)
        self.title("Nouvelle Partie")
        self.configure(bg='black')

        # Frame principal pour tout contenir
        frame_principal = Frame(self, bg="burlywood", padx=10, pady=10)
        frame_principal.pack(fill=BOTH, expand=True)

        # Frame pour les boutons et contrôles à gauche
        frame_gauche = Frame(frame_principal, bg="burlywood", padx=10)
        frame_gauche.pack(side=LEFT, fill=Y)

        # Bouton retour au menu
        button_back = Button(frame_gauche, text="Retour au menu", command=self.retour_menu,
                           bg="saddlebrown", fg="white", font=("Arial", 12, "bold"))
        button_back.pack(pady=10)

        # Séparateur
        Frame(frame_gauche, height=2, bg="saddlebrown").pack(fill=X, pady=5)

        # Label et Entry pour les itérations MCTS
        Label(frame_gauche, text="Itérations MCTS:", font=("Arial", 12), bg="burlywood").pack(pady=5)
        self.__iterations_var = IntVar(value=self.__mcts_iterations)
        self.__iterations_entry = Entry(frame_gauche, textvariable=self.__iterations_var, width=10)
        self.__iterations_entry.pack(pady=5)
        
        # Bouton pour appliquer le changement d'itérations
        Button(frame_gauche, text="Appliquer", command=self.__update_iterations,
               bg="saddlebrown", fg="white", font=("Arial", 10)).pack(pady=5)

        # Checkbox pour la colormap
        self.__show_colormap = BooleanVar(value=False)
        Checkbutton(frame_gauche, text="Afficher colormap", 
                   variable=self.__show_colormap,
                   bg="burlywood",
                   font=("Arial", 12)).pack(pady=10)

        # Frame central pour contenir le plateau et les boutons du haut
        frame_central = Frame(frame_principal, bg="burlywood")
        frame_central.pack(side=LEFT, fill=BOTH, expand=True)

        # Frame pour les boutons du haut
        frame_boutons = Frame(frame_central, bg="burlywood")
        frame_boutons.pack(pady=10)

        # Bouton compter les points
        button_score = Button(frame_boutons, text="Compter points", command=self.toogle_compter_points,
                            bg="saddlebrown", fg="white", font=("Arial", 12, "bold"))
        button_score.grid(row=0, column=1, padx=10, pady=5)

        # Bouton personnalisé
        button_perso = Button(frame_boutons, text="Perso", command=self.toogle_perso,
                            bg="saddlebrown", fg="white", font=("Arial", 12, "bold"))
        button_perso.grid(row=1, column=1, padx=10, pady=5)

        # Label pour afficher la taille du Goban
        label_info = Label(frame_boutons, text=f"Taille du Goban : {self.__size}x{self.__size}", 
                          font=("Arial", 14, "bold"), bg="burlywood", fg="black", width=30)
        label_info.grid(row=0, column=0, padx=10, pady=5)

        # Frame pour le canvas avec une taille minimale
        frame_canvas = Frame(frame_central, bg="burlywood")
        frame_canvas.pack(expand=True, fill=BOTH)

        # Création du canvas
        self.__canvas_goban = Canvas(frame_canvas, width=self.__goban_size, height=self.__goban_size, 
                                   bg="burlywood", borderwidth=2, relief="ridge")
        self.__canvas_goban.pack()

        # Initialisation du moteur de jeu
        self.__init_moteur_de_jeu()

        # Si c'est l'IA qui commence, jouer le premier coup
        if self.__ai and self.__ai_player == BLACK:
            self.__moteurDeJeu.jouer_premier_coup()
        elif self.__ai_vs_ai:
            self.__moteurDeJeu.jouer_automatiquement()

        # Dessiner le plateau
        self.dessiner_plateau()
        self.bind_events()

    def __init_moteur_de_jeu(self):
        """Initialise le moteur de jeu en fonction des paramètres"""
        if self.__ai_vs_ai:
            self.__moteurDeJeu = MoteurDeJeuAIvsAI(self, self.__ai, self.__ai_player, self.__size, self.__sbx, self.__mcts_iterations, self.__menuGraphique.ia_vs_ia_config)
        elif self.__cnn:
            self.__moteurDeJeu = MoteurDeJeuCNN(self, self.__ai, self.__ai_player, self.__size, self.__sbx, self.__mcts_iterations)
        else:
            self.__moteurDeJeu = MoteurDeJeu(self, self.__ai, self.__ai_player, self.__size, self.__sbx, self.__mcts_iterations)

    def get_canvas(self):
        return self.__canvas_goban

    def toogle_perso(self):
        print(self.__moteurDeJeu.get_plateau().group_in_enemy_territory((2, 2), WHITE))

    def toogle_compter_points(self):
        self.dessiner_territoires()
        self.__moteurDeJeu.compter_points()

    def toogle_jouer_coup(self, coup):
        self.__moteurDeJeu.jouer_coup(coup)

    def on_enter(self):
        if self.__preview_oval is not None:
            self.__canvas_goban.delete(self.__preview_oval)
        self.__preview = True
    
    def on_leave(self):
        if self.__preview_oval is not None:
            self.__canvas_goban.delete(self.__preview_oval)
        self.__preview = False

    def retour_menu(self):
        self.destroy()
        print("Retour au menu principal")
        self.__menuGraphique.reset()
        self.__menuGraphique.deiconify()

    def get_coord(self,event):
        # Convertir les coordonnées du clic en index d'intersection
        x, y = round((event.x-self.__cell_size) / self.__cell_size), round((event.y-self.__cell_size) / self.__cell_size)
        return (y,x)  # Retourner dans l'ordre (y,x) pour correspondre à l'ordre du plateau

    def update_affichage(self):
        
        self.__canvas_goban.delete("stone")

        for x in range(self.__size):

            for y in range(self.__size):

                joueur = self.__moteurDeJeu.get_plateau().get_plateau()[y,x]

                if joueur!=EMPTY:
                    color = 'white' if joueur == WHITE else 'black'
                    
                    # Dessiner la pierre sur l'intersection
                    self.__canvas_goban.create_oval(
                        self.__cell_size + (x * self.__cell_size - (self.__cell_size/2)),
                        self.__cell_size + (y * self.__cell_size - (self.__cell_size/2)),
                        self.__cell_size + (x * self.__cell_size + (self.__cell_size/2)),
                        self.__cell_size + (y * self.__cell_size + (self.__cell_size/2)),
                        fill=color,
                        outline="black",
                        tags="stone")

    def keypressed(self, event):
        if self.__sbx:
            if event.keysym == "b":
                print("Changement de joueur...")
                self.__moteurDeJeu.changer_joueur()
                self.update_affichage()
        if event.keysym == "g":
            self.__moteurDeJeu.get_plateau().get_groupes_morts()

    def preview_stone(self, event):
        """Affiche une prévisualisation dynamique de la pierre."""
        
        if not self.__preview:
            return
        self.__canvas_goban.delete("territoires")
        
        # Convertir les coordonnées du curseur en index d'intersection
        x, y = round((event.x - self.__cell_size) / self.__cell_size), round((event.y-self.__cell_size) / self.__cell_size)

        if 0 <= x < self.__size and 0 <= y < self.__size:
            # Effacer la pierre de prévisualisation précédente
            if self.__preview_oval is not None:
                self.__canvas_goban.delete(self.__preview_oval)

            color = "black" if self.__moteurDeJeu.get_turn() == BLACK else "white"

            # Dessiner la prévisualisation sur l'intersection
            self.__preview_oval = self.__canvas_goban.create_oval(
                self.__cell_size + (x * self.__cell_size - (self.__cell_size/2)),
                self.__cell_size + (y * self.__cell_size - (self.__cell_size/2)),
                self.__cell_size + (x * self.__cell_size + (self.__cell_size/2)),
                self.__cell_size + (y * self.__cell_size + (self.__cell_size/2)),
                fill=color,
                outline="gray"
            )

    def dessiner_territoires(self):

        self.__canvas_goban.delete("territoires")

        color = ['green', 'red']
        
        #ter_noir, ter_blanc = self.get_moteurDeJeu().get_plateau().find_territoires()
        #morts_noir, morts_blanc = self.get_moteurDeJeu().get_plateau().get_groupes_morts()

        #print("morts noirs:", ter_noir)
        #print("morts blancs:", ter_blanc)
        #print()


        for i, territoires in enumerate(self.__moteurDeJeu.get_plateau().find_territoires()):
            c = color[i]
            
            for territoire in territoires:
                for coord in territoire:
                    
                    y,x =  coord[0], coord[1]  # coord est déjà dans l'ordre (y,x)
                    d = 2.5
                    self.__canvas_goban.create_oval(
                            self.__cell_size + (x * self.__cell_size - (self.__cell_size/8))+(-1)**(i)*d,
                            self.__cell_size + (y * self.__cell_size - (self.__cell_size/8)),
                            self.__cell_size + (x * self.__cell_size + (self.__cell_size/8)+(-1)**(i)*d),
                            self.__cell_size + (y * self.__cell_size + (self.__cell_size/8)),
                            fill=c,
                            outline=c,
                            tags="territoires")

    def __update_iterations(self):
        """Met à jour le nombre d'itérations MCTS"""
        try:
            new_iterations = self.__iterations_var.get()
            if new_iterations > 0:
                self.__mcts_iterations = new_iterations
                if hasattr(self, '_GobanGraphique__moteurDeJeu'):
                    self.__moteurDeJeu.set_mcts_iterations(new_iterations)
        except:
            pass  # Ignorer les valeurs invalides

    def get_show_colormap(self):
        """Retourne si la colormap doit être affichée"""
        return self.__show_colormap.get()

    def dessiner_plateau(self):
        """Dessine le plateau de jeu"""
        # Dessiner la grille
        for i in range(self.__size):
            self.__canvas_goban.create_line(
                self.__cell_size + self.__cell_size * i, self.__cell_size, 
                self.__cell_size + self.__cell_size * i, 
                self.__cell_size * (self.__size - 1) + self.__cell_size, 
                fill="black"
            )
            self.__canvas_goban.create_line(
                self.__cell_size, self.__cell_size + self.__cell_size * i, 
                self.__cell_size * (self.__size - 1) + self.__cell_size, 
                self.__cell_size + self.__cell_size * i, 
                fill="black"
            )

        # Déterminer les positions des hoshis
        hoshis_positions = []
        if self.__size == 19:
            hoshis_positions = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif self.__size == 13:
            hoshis_positions = [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
        elif self.__size == 9:
            hoshis_positions = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        elif self.__size == 5:
            hoshis_positions = [(2, 2)]

        # Dessiner les hoshis
        for x, y in hoshis_positions:
            self.__canvas_goban.create_oval(
                self.__cell_size + (x * self.__cell_size - (self.__cell_size / 8)),
                self.__cell_size + (y * self.__cell_size - (self.__cell_size / 8)),
                self.__cell_size + (x * self.__cell_size + (self.__cell_size / 8)),
                self.__cell_size + (y * self.__cell_size + (self.__cell_size / 8)),
                fill='black',
                outline="black",
            )

    def bind_events(self):
        """Lie les événements aux fonctions correspondantes"""
        # Ajout des événements
        self.__canvas_goban.bind("<Button-1>", lambda event: self.toogle_jouer_coup(self.get_coord(event)))
        self.__canvas_goban.bind("<KeyPress>", lambda event: self.keypressed(event))
        self.__canvas_goban.focus_set()
        self.__canvas_goban.bind('<Button-3>', lambda event: self.toogle_jouer_coup(None))
        self.__canvas_goban.bind("<Motion>", lambda event: self.preview_stone(event))
        self.__canvas_goban.bind("<Enter>", lambda event: self.on_enter())
        self.__canvas_goban.bind("<Leave>", lambda event: self.on_leave())

class MenuGraphique(Tk):

    def __init__(self):
        self.__ai = False
        self.__ai_player = None
        self.__size = None
        self.__sbx = False
        self.__ai_vs_ai = False
        self.__cnn = False
        self.ia_vs_ia_config = None
        Tk.__init__(self)
        self.title("Jeu de Go")
        
        # Charger l'image de fond
        self.img = PhotoImage(file="jeu_de_go/ressource/background.png")
        background_label = Label(self, image=self.img)
        background_label.place(relwidth=1, relheight=1)  # Remplit toute la fenêtre
        
        # Cadre principal
        frame = Frame(self, bg="burlywood", padx=20, pady=20)
        frame.pack(pady=20, padx=20)
        
        # Titre du menu
        label_title = Label(frame, text="Lancer une partie", font=("Arial", 20, "bold"), bg="burlywood", fg="black")
        label_title.pack(pady=10)
        
        # Sélection de la taille du Goban
        self.__label_size = Label(frame, text="Choisissez la taille du Goban :", font=("Arial", 14, "bold"), width=30, anchor='center', bg="burlywood", fg="black")
        self.__label_size.pack(pady=5)

        self.__size = None

        def set_size(size):
            self.__size = size
            self.__label_size.config(text=f"Taille du Goban sélectionnée : {size}")
            self.__bouton_sandbox.config(state="normal")
            self.__bouton_split.config(state="normal")
            self.__bouton_ai_player_begins.config(state="normal")
            self.__bouton_ai_player_not_begins.config(state="normal")
            self.__bouton_ai_vs_ai.config(state="normal")

        # Création d'un sous-frame pour organiser les boutons en grille
        frame_buttons = Frame(frame, bg="burlywood")
        frame_buttons.pack(pady=10)

        sizes = [5, 9, 13, 19]
        for i, size in enumerate(sizes):
            row, col = divmod(i, 2)  # Calcule la ligne et la colonne (2 colonnes max)
            bouton_size = Button(frame_buttons, text=f"{size}x{size}", width=10, height=1, relief="groove",
                                    font=("Helvetica", 12, "bold"), command=lambda s=size: set_size(s))
            bouton_size.grid(row=row, column=col, padx=10, pady=5)
        
        # Choix du mode de jeu
        label_mode = Label(frame, text="Choisissez le mode de jeu :", font=("Arial", 14, "bold"), bg="burlywood", fg="black")
        label_mode.pack(pady=10)
        
        self.open_game_window = self._open_game_window
        
        self.__bouton_sandbox = Button(frame, text="Sandbox", width=20, height=2, relief="groove", font=("Helvetica", 12, "bold"), state='disabled', command=lambda: self.open_game_window(self.sbx_begin))
        self.__bouton_split = Button(frame, text="2 joueurs", width=20, height=2, relief="groove", font=("Helvetica", 12, "bold"), state='disabled', command=lambda: self.open_game_window())
        self.__bouton_ai_player_begins = Button(frame, text="Contre l'IA en noir", width=20, height=2, relief="groove", font=("Helvetica", 12, "bold"), state='disabled', command=lambda: self.open_game_window(self.ai_initialize_player_begins))
        self.__bouton_ai_player_not_begins = Button(frame, text="Contre l'IA en blanc", width=20, height=2, relief="groove", font=("Helvetica", 12, "bold"), state='disabled', command=lambda: self.open_game_window(self.ai_initialize_player_not_begins))
        self.__bouton_ai_vs_ai = Button(frame, text="AI vs AI", width=20, height=2, relief="groove", font=("Helvetica", 12, "bold"), state='disabled', command=self.config_ia_vs_ia)
        
        self.__bouton_sandbox.pack(pady=5)
        self.__bouton_split.pack(pady=5)
        self.__bouton_ai_player_begins.pack(pady=5)
        self.__bouton_ai_player_not_begins.pack(pady=5)
        self.__bouton_ai_vs_ai.pack(pady=5)

        # Bouton quitter
        button_quit = Button(frame, text="Quitter", command=self.destroy, bg='burlywood', font=("Helvetica", 12, "bold"))
        button_quit.pack(pady=10)

    def reset(self):
        self.__ai = False
        self.__ai_player = None
        self.__sbx = False
        self.__ai_vs_ai = False
        self.__cnn = False

    def quitter(self, fenetre):
        fenetre.destroy()  # Fermer la fenêtre

    def ai_vs_ai_initialize(self):
        self.__ai_vs_ai = True

    def ai_initialize_player_begins(self):
        self.__ai = True
        self.__ai_player = WHITE

    def ai_initialize_player_not_begins(self):
        self.__ai = True
        self.__ai_player = BLACK

    def sbx_begin(self):
        self.__sbx = True
        
    def config_ia_vs_ia(self):
        config_win = Toplevel(self)
        config_win.title("Configuration IA vs IA")

        Label(config_win, text="Type IA Noir:").grid(row=0, column=0)
        ia_noir_var = StringVar(value="mcts_classique")
        OptionMenu(config_win, ia_noir_var, "mcts_classique", "neuralnet").grid(row=0, column=1)

        Label(config_win, text="Itérations Noir:").grid(row=1, column=0)
        iter_noir_var = IntVar(value=800)
        Entry(config_win, textvariable=iter_noir_var).grid(row=1, column=1)

        Label(config_win, text="Modèle Policy Noir:").grid(row=2, column=0)
        policy_noir_var = StringVar(value=r"katago_relatives\katago-opencl-linux\networks\training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed\best_policynet_9x9.pth")
        Entry(config_win, textvariable=policy_noir_var).grid(row=2, column=1)

        Label(config_win, text="Modèle Value Noir:").grid(row=3, column=0)
        value_noir_var = StringVar(value=r"katago_relatives\katago-opencl-linux\networks\training_logs_value9x9_v3_2025-06-06_18-21-00\best_valuenet_9x9.pth")
        Entry(config_win, textvariable=value_noir_var).grid(row=3, column=1)

        use_tqdm_noir_var = BooleanVar(value=False)
        Checkbutton(config_win, text="Afficher barre de progression (Noir)", variable=use_tqdm_noir_var).grid(row=4, column=0, columnspan=2)
        autorise_pass_noir_var = BooleanVar(value=True)
        Checkbutton(config_win, text="Autoriser Pass (Noir)", variable=autorise_pass_noir_var).grid(row=5, column=0, columnspan=2)

        Label(config_win, text="Type IA Blanc:").grid(row=6, column=0)
        ia_blanc_var = StringVar(value="mcts_classique")
        OptionMenu(config_win, ia_blanc_var, "mcts_classique", "neuralnet").grid(row=6, column=1)

        Label(config_win, text="Itérations Blanc:").grid(row=7, column=0)
        iter_blanc_var = IntVar(value=800)
        Entry(config_win, textvariable=iter_blanc_var).grid(row=7, column=1)

        Label(config_win, text="Modèle Policy Blanc:").grid(row=8, column=0)
        policy_blanc_var = StringVar(value=r"katago_relatives\katago-opencl-linux\networks\training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed\best_policynet_9x9.pth")
        Entry(config_win, textvariable=policy_blanc_var).grid(row=8, column=1)

        Label(config_win, text="Modèle Value Blanc:").grid(row=9, column=0)
        value_blanc_var = StringVar(value=r"katago_relatives\katago-opencl-linux\networks\training_logs_value9x9_v3_2025-06-06_18-21-00\best_valuenet_9x9.pth")
        Entry(config_win, textvariable=value_blanc_var).grid(row=9, column=1)

        use_tqdm_blanc_var = BooleanVar(value=False)
        Checkbutton(config_win, text="Afficher barre de progression (Blanc)", variable=use_tqdm_blanc_var).grid(row=10, column=0, columnspan=2)
        autorise_pass_blanc_var = BooleanVar(value=True)
        Checkbutton(config_win, text="Autoriser Pass (Blanc)", variable=autorise_pass_blanc_var).grid(row=11, column=0, columnspan=2)

        def valider():
            self.ia_vs_ia_config = {
                "noir": {
                    "type": ia_noir_var.get(),
                    "iterations": iter_noir_var.get(),
                    "policy": policy_noir_var.get(),
                    "value": value_noir_var.get(),
                    "use_tqdm": use_tqdm_noir_var.get(),
                    "autorise_pass": autorise_pass_noir_var.get()
                },
                "blanc": {
                    "type": ia_blanc_var.get(),
                    "iterations": iter_blanc_var.get(),
                    "policy": policy_blanc_var.get(),
                    "value": value_blanc_var.get(),
                    "use_tqdm": use_tqdm_blanc_var.get(),
                    "autorise_pass": autorise_pass_blanc_var.get()
                }
            }
            config_win.destroy()
            self.open_game_window(self.ai_vs_ai_initialize)

        Button(config_win, text="Valider", command=valider).grid(row=12, column=0, columnspan=2)

    def _open_game_window(self, mode_function=None):
        if self.__size is None:
            self.__label_size.config(text="Choisissez la taille du Goban :")
            return
        self.withdraw()
        if mode_function:
            mode_function()
        if hasattr(self, '_MenuGraphique__cnn') and self.__cnn:
            self.__fenetreGoban = GobanGraphique(self, True, self.__ai_player, self.__size, False, False, True, 2500)
        else:
            self.__fenetreGoban = GobanGraphique(self, self.__ai, self.__ai_player, self.__size, self.__sbx, self.__ai_vs_ai, False, 2500)

if __name__ == "__main__":
    Go = MenuGraphique()
    Go.mainloop()
