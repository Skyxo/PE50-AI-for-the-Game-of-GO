# PE50-JeuDeGo

## Membres du projet
Sacha Vanderaspoilden

Théo Huangfu

Charles Bergeat

Lou Thomas

Timothée Mizon

Rodrigue Miloud

## Tuteur
Philippe Michel

Alexandre Saidi

Abdel-Malik Zine

coucou
## Conseiller 
Mhosen Ardibilian


## Accéder au serveur gpu
1) Être connecté au wifi de Centrale Lyon
2) Ouvrir Windows PowerShell et entrer
idf : ssh projet20@156.18.90.98
mdp : ******
3) Pour visualiser les fichiers à l'intérieur du répertoire, taper "ls"
4) Pour se déplacer, taper "cd [nom du dossier]"
5) "cd .." pour retourner en arrière

## COMMANDES LINUS ##
Commandes Linux
ls : liste les fichiers et dossiers dans le répertoire courant
cd [nom du dossier] : change de répertoire
pwd : affiche le chemin du répertoire courant
mkdir [nom du dossier] : crée un nouveau dossier
rm [nom du fichier] : supprime un fichier
rm -r [nom du dossier] : supprime un dossier et son contenu
cp [fichier source] [fichier destination] : copie un fichier
mv [fichier source] [fichier destination] : déplace ou renomme un fichier
touch [nom du fichier] : crée un nouveau fichier
cat [nom du fichier] : affiche le contenu d'un fichier
nano [nom du fichier] : édite un fichier avec l'éditeur de texte Nano
chmod [permissions] [nom du fichier] : change les permissions d'un fichier
grep [motif] [nom du fichier] : recherche un motif dans un fichier
find [répertoire] -name [nom du fichier] : recherche un fichier dans un répertoire
tar -xvf [nom du fichier.tar] : extrait un fichier tar
tar -cvf [nom du fichier.tar] [fichiers à archiver] : crée une archive tar
ssh [utilisateur]@[adresse IP] : se connecte à un serveur distant via SSH
scp [fichier source] [utilisateur]@[adresse IP]:[fichier destination] : copie un fichier vers un serveur distant via SCP
man [commande] : affiche le manuel d'une commande

## COMMANDES GITHUB ##
Pour fet origin, taper : "git fetch origin" et entrer ses idf/mdp de la DSI ECL
Pour pull, taper : "git pull" et entrer ses idf/mdp de la DSI ECL
Pour changer de branche : git switch [nom de la branche]
# 1. Ajouter les fichiers modifiés
git add training.py  # ou . pour tout
# 2. Commit
git commit -m "Ajout du logging et des checkpoints"
# 3. Push vers le dépôt distant
git push origin nom_de_la_branche
# 4. Safe push
git pull origin testcharles_ameliore --rebase
puis :
git push origin testcharles_ameliore

## SESSION TMUX ##
1) Créer une nouvelle session nommée training : tmux new -s training
2) Lancer ton script d'entraînement dedans : python3 training.py --data fox_dataset.npz --bs 512 --lr 1e-3 --epochs 20
3) Détacher proprement la session (laisser tourner en fond) : Ctrl + B puis D (D comme “detach”)
Tu vois quelque chose comme : [detached (from session training)] -> Tu es revenu dans le terminal normal, mais ton entraînement continue en arrière-plan dans la session tmux.
4) Rejoindre la session plus tard (même après déconnexion SSH) : tmux attach -t training
-> Tu retrouves ton terminal tel que tu l’avais laissé (permet de suivre la progression ou relancer un script à la main).
5) Fermer une session tmux proprement : exit
6) Liste des sessions tmux existantes : tmux ls (Tu verras par exemple : training: 1 windows (created Mon 12:00) [80x24])

7) Arrêter une session tmux : tmux kill-session -t ma_session

8) Arrêter la session dès que l'entraînement est terminé : 
tmux new -s training "python training.py --data fox_dataset.npz --epochs 20 && tmux kill-session -t training"

9) Copier toute la console : 

## LANCER L'ENTRAÎNEMENT ##
1) Se mettre dans le même répertoire que training.py
2) envoyer : python3 training.py --data fox_dataset.npz --bs 512 --lr 1e-3 --epochs 20
