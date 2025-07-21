import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

# Dataset pour policy network 9x9
class GoPolicyDataset9x9(Dataset):
    def __init__(self, npz_path=None, augment=False):
        data = np.load(npz_path)
        self.X = data['X'].astype('float32')  # (N, 2, 9, 9)
        self.y = data['policy_targets'].astype('float32')  # (N, 82)
        self.augment = augment
        bs = self.X.shape[2]
        self.bs = bs
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx].copy()
        
        if self.augment:
            # Déterminer la transformation aléatoire
            transform_type = np.random.randint(0, 8)
            k_rot = transform_type % 4  # Nombre de rotations de 90°
            do_flip = transform_type >= 4  # Si True, on fait un flip horizontal
            
            # Appliquer les rotations
            if k_rot > 0:
                # Rotation de l'entrée (canaux, hauteur, largeur)
                x = np.rot90(x, k=k_rot, axes=(1, 2))
                
                # Rotation des cibles (positions sur le plateau)
                y_board = y[:-1].reshape(self.bs, self.bs)
                y_pass = y[-1]  # Sauvegarder la probabilité de "pass"
                
                # Appliquer la même rotation aux cibles
                y_board = np.rot90(y_board, k=k_rot)
                
                # Remettre à plat les cibles du plateau
                y[:-1] = y_board.flatten()
                y[-1] = y_pass  # Restaurer la probabilité de "pass"
            
            # Appliquer le flip horizontal (si nécessaire)
            if do_flip:
                # Flip horizontal de l'entrée (miroir sur l'axe vertical)
                x = np.flip(x, axis=2)  # Flip sur la dimension largeur
                
                # Flip horizontal des cibles
                y_board = y[:-1].reshape(self.bs, self.bs)
                y_pass = y[-1]
                
                # Appliquer le même flip horizontal aux cibles
                y_board = np.fliplr(y_board)  # Flip left-right (horizontal)
                
                # Remettre à plat les cibles du plateau
                y[:-1] = y_board.flatten()
                y[-1] = y_pass  # La probabilité de "pass" ne change pas
        
        # Conversion en tenseurs PyTorch
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        
        return x, y

# PolicyNet (tronc commun + policy head) - Version optimisée
class PolicyNet9x9(nn.Module):
    def __init__(self, board_size=9, n_blocks=8, filters=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Pile de blocs résiduels améliorés (réduite pour éviter le surapprentissage)
        self.resblocks = nn.Sequential(*[EnhancedResidualBlock(filters) for _ in range(n_blocks)])
        
        # Branche d'attention globale
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Tête de politique optimisée
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, filters, 1, bias=False),  # Réduit la complexité
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # Dropout modéré
            nn.Conv2d(filters, filters // 2, 1, bias=False),  # Réduction progressive
            nn.BatchNorm2d(filters // 2),
            nn.ReLU(inplace=True)
        )
        
        self.policy_fc = nn.Linear((filters // 2) * board_size * board_size, board_size * board_size + 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.resblocks(x)
        
        # Appliquer l'attention globale
        attention = self.global_attention(x)
        x = x * attention
        
        x = self.policy_head(x)
        x = x.view(x.size(0), -1)
        policy_logits = self.policy_fc(x)
        return policy_logits

class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation block pour attention par canal
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Appliquer SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        return F.relu(out + res)

class EarlyStopper:
    def __init__(self, patience=100, min_delta=1e-4, base_min_lr_factor=10):
        self.patience = patience
        self.base_min_delta = min_delta  # Valeur de base
        self.min_delta = min_delta       # Valeur actuelle (potentiellement ajustée)
        self.base_min_lr_factor = base_min_lr_factor
        self.min_lr_factor = base_min_lr_factor  # Sera dynamiquement ajusté
        self.best_loss = math.inf
        self.counter = 0
        
    def adjust_for_lr(self, current_lr):
        """Ajuste min_delta et min_lr_factor en fonction du learning rate actuel"""
        # Ajuster dynamiquement le min_lr_factor selon le LR
        if current_lr < 1e-9:
            # LR ultra-faible (< 1e-9): facteur très élevé
            self.min_lr_factor = 10000
        elif current_lr < 1e-8:
            # LR très faible (< 1e-8): facteur élevé
            self.min_lr_factor = 5000
        elif current_lr < 1e-7:
            # LR faible (< 1e-7): facteur plus élevé
            self.min_lr_factor = 1000
        elif current_lr < 1e-6:
            # LR assez faible (< 1e-6): facteur modérément élevé
            self.min_lr_factor = 500
        elif current_lr < 1e-5:
            # LR moyen-faible (< 1e-5): facteur légèrement élevé
            self.min_lr_factor = 100
        else:
            # LR normal: facteur standard
            self.min_lr_factor = self.base_min_lr_factor
        
        # Calculer le min_delta avec le nouveau facteur
        if current_lr < self.base_min_delta / self.min_lr_factor:
            self.min_delta = current_lr * self.min_lr_factor
        else:
            self.min_delta = self.base_min_delta
            
        print(f"Adjusted min_lr_factor: {self.min_lr_factor}, min_delta: {self.min_delta:.10e}")
        return self.min_delta
        
    def step(self, loss):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def refresh_model(net, optimizer, device):
    """
    Effectue un rafraîchissement du modèle pour améliorer les performances.
    
    Cette technique sauvegarde et recharge le modèle, ce qui a pour effet:
    - De réinitialiser partiellement les états internes de l'optimiseur
    - De stabiliser les poids du modèle pour améliorer la convergence
    - De fournir un nouvel élan d'apprentissage en sortant des plateaux
    
    Args:
        net: Le modèle à rafraîchir
        optimizer: L'optimiseur associé au modèle
        device: L'appareil sur lequel le modèle est chargé
    
    Returns:
        tuple: (modèle rafraîchi, optimiseur modifié)
    """
    print("🔄 Performing model refreshing...")
    
    # 1. Sauvegarder les poids du modèle dans un fichier temporaire
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refresh_tmp_policy.pth')
    torch.save(net.state_dict(), tmp_path)
    
    # 2. Réinitialiser partiellement l'optimiseur (réduire le momentum)
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            param_state = optimizer.state.get(p)
            if param_state is not None and 'momentum_buffer' in param_state:
                # Réduire le momentum de moitié pour permettre de nouveaux chemins d'optimisation
                param_state['momentum_buffer'] *= 0.5
    
    # 3. Recharger le modèle
    state_dict = torch.load(tmp_path)
    net.load_state_dict(state_dict)
    
    # 4. Nettoyer le fichier temporaire
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    print("✅ Model refreshed successfully")
    return net, optimizer

# Updated loss function to KL divergence
def policy_loss(logits, targets):
    """KL divergence loss pour l'apprentissage de la distribution de policy"""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss

# Policy Loss class with label smoothing for regularization
class PolicyLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super(PolicyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        KL divergence loss avec label smoothing pour l'apprentissage de la distribution de policy
        """
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            # Create a smooth distribution
            n_classes = targets.size(1)
            uniform_distribution = torch.ones_like(targets) / n_classes
            smooth_targets = (1 - self.label_smoothing) * targets + self.label_smoothing * uniform_distribution
            targets = smooth_targets
            
        # Calculate KL divergence
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss

# Legacy function for compatibility with label smoothing
def policy_loss_with_smoothing(logits, targets, label_smoothing=0.1):
    """KL divergence loss pour l'apprentissage de la distribution de policy avec label smoothing"""
    loss_fn = PolicyLoss(label_smoothing)
    return loss_fn(logits, targets)

def extract_checkpoint_info(checkpoint_path):
    """
    Extrait les informations d'un checkpoint: époque et learning rate
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        
    Returns:
        tuple: (epoch, learning_rate)
            - epoch: Dernière époque enregistrée dans le checkpoint
            - learning_rate: Dernier learning rate utilisé
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Si c'est un état du modèle complet (avec des métadonnées)
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get('epoch', None)
            learning_rate = checkpoint.get('learning_rate', None)
            
            if epoch is not None:
                print(f"Trouvé epoch {epoch} dans le checkpoint")
            if learning_rate is not None:
                print(f"Trouvé learning rate {learning_rate} dans le checkpoint")
                
            return epoch, learning_rate
        
        # Si c'est seulement un modèle sans métadonnées
        print("Aucune métadonnée trouvée dans le checkpoint")
        return None, None
        
    except Exception as e:
        print(f"Erreur lors de la lecture du checkpoint: {e}")
        return None, None

def load_metrics_from_csv(csv_path):
    """
    Charge les métriques existantes depuis un fichier CSV
    
    Returns:
        tuple: (headers, metrics, last_epoch, last_lr)
            - headers: Liste des en-têtes de colonnes
            - metrics: Liste de toutes les lignes de métriques
            - last_epoch: Dernière époque enregistrée
            - last_lr: Dernier learning rate utilisé
    """
    if not os.path.exists(csv_path):
        return None, [], 0, None
    
    metrics = []
    headers = None
    last_lr = None
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Lire les en-têtes
        
        # Trouver l'index de la colonne Learning Rate
        lr_index = None
        for i, header in enumerate(headers):
            if header == 'Learning Rate':
                lr_index = i
                break
        
        for row in reader:
            metrics.append(row)
            # Extraire le learning rate de la dernière ligne
            if lr_index is not None and len(row) > lr_index:
                try:
                    last_lr = float(row[lr_index])
                except ValueError:
                    pass
    
    if not metrics:
        return headers, [], 0, None
    
    # Obtenir la dernière époque
    last_epoch = int(metrics[-1][0])
    
    return headers, metrics, last_epoch, last_lr


def plot_training_metrics(csv_path, output_dir, csv_headers):
    """
    Fonction dédiée pour tracer les graphiques d'entraînement de manière robuste
    """
    all_data = {header: [] for header in csv_headers}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                for header in csv_headers:
                    if header in line and line[header].strip():
                        try:
                            all_data[header].append(float(line[header]))
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}")
        return
    
    # Vérifier qu'on a des données à tracer
    if not all_data['Epoch']:
        print("Pas de données à tracer")
        return
        
    plot_config = [
        (['Train Loss', 'Val Loss'], None, 'Loss')
    ]
    
    # N'ajouter l'entropie que si des données sont disponibles
    if all_data['Pred Entropy'] and all_data['Target Entropy']:
        plot_config.append((['Pred Entropy', 'Target Entropy'], None, 'Entropy (bits)'))
    
    plt.figure(figsize=(15, 5 * len(plot_config)))
    
    for i, (metrics, ylim, title) in enumerate(plot_config):
        plt.subplot(len(plot_config), 1, i+1)
        
        # Obtenir le plus grand nombre d'époques disponible
        epochs_data = range(1, len(all_data['Epoch']) + 1)
        
        for metric in metrics:
            if metric in all_data and all_data[metric]:
                metric_data = all_data[metric]
                
                # Si cette métrique a moins de points que d'époques
                if len(metric_data) < len(epochs_data):
                    # Tracer uniquement sur les dernières époques disponibles
                    offset = len(epochs_data) - len(metric_data)
                    plt.plot(epochs_data[offset:], metric_data, 
                             label=f"{metric} (dernières {len(metric_data)} époques)")
                else:
                    plt.plot(epochs_data[:len(metric_data)], metric_data, label=metric)
                    
        plt.title(title)
        plt.xlabel('Epochs')
        if ylim:
            plt.ylim(ylim)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    plt.close()


def train_policy_9x9(npz_path, batch_size=128, lr=5e-4, epochs=200, val_split=0.1, device=None, 
                     label_smoothing=0.1, resume_from=None, continue_output_dir=None, ultra_low_lr=False,
                     existing_metrics=None, start_epoch=1, last_lr=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Définir le répertoire de sortie
    if continue_output_dir:
        output_dir = continue_output_dir
        print(f"Using output directory: {output_dir}")
    else:
        timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'training_logs_policy9x9_v5_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Starting new training run at: {timestamp}")
    
    # Chargement des métriques existantes si on continue un entraînement
    csv_headers = ['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate', 'Pred Entropy', 'Target Entropy']
    csv_path = os.path.join(output_dir, 'metrics_all.csv')
    
    # Utiliser les métriques existantes si elles sont fournies, sinon les charger
    if existing_metrics is None:
        existing_metrics = []
    
    ds = GoPolicyDataset9x9(npz_path, augment=True)
    
    # Analyser l'entropie des distributions cibles
    def calculate_entropy(probs):
        """Calcule l'entropie de Shannon pour des distributions de probabilités"""
        return -np.sum(probs * np.log2(probs + 1e-10), axis=1)

    # Calculer les statistiques d'entropie sur l'ensemble du dataset
    all_targets = ds.y  # Shape: (N, 82)
    entropies = calculate_entropy(all_targets)
    mean_entropy = entropies.mean()
    median_entropy = np.median(entropies)
    min_entropy = entropies.min()
    max_entropy = entropies.max()

    print(f"\nStatistiques d'entropie des cibles:")
    print(f"  Moyenne: {mean_entropy:.4f} bits")
    print(f"  Médiane: {median_entropy:.4f} bits")
    print(f"  Min: {min_entropy:.4f} bits")
    print(f"  Max: {max_entropy:.4f} bits")
    print(f"  Théorique max pour {all_targets.shape[1]} classes: {np.log2(all_targets.shape[1]):.4f} bits")

    n_val = int(len(ds)*val_split)
    n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    
    # Ajuster les paramètres du DataLoader pour un entraînement très long
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),  # Utiliser tous les cœurs disponibles
        persistent_workers=True,  # Évite de recréer les workers
        drop_last=True  # Stabilise l'entraînement avec des batches de taille constante
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    
    # Créer ou charger le modèle
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from}")
        checkpoint_data = torch.load(resume_from)
        
        # Vérifier si c'est un dictionnaire d'état ou un checkpoint complet
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['model_state_dict']
        else:
            model_state_dict = checkpoint_data
            
        # Créer l'instance du modèle et charger les poids
        net = PolicyNet9x9().to(device)
        print("Loading model state dict...")
        # Vérifier le format du modèle et adapter si nécessaire
        keys = list(model_state_dict.keys())
        if len(keys) > 0:
            if keys[0].startswith('module.') and not isinstance(net, torch.nn.DataParallel):
                print("Converting DataParallel model to single-device model")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # remove 'module.'
                    new_state_dict[name] = v
                model_state_dict = new_state_dict
            
            # Si toujours des problèmes, essayer le chargement non-strict
            try:
                net.load_state_dict(model_state_dict)
            except:
                print("Strict loading failed, attempting non-strict loading")
                net.load_state_dict(model_state_dict, strict=False)

        # Maintenant on peut charger le state_dict
        net.load_state_dict(model_state_dict)

        net.load_state_dict(model_state_dict)
        print("Model loaded successfully")
    else:
        net = PolicyNet9x9().to(device)
        
    # Check for multiple GPUs and wrap the model with DataParallel if available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = torch.nn.DataParallel(net)
    else:
        print("Using a single GPU or CPU")
    
    # Déterminer le learning rate initial
    initial_lr = lr
    
    # Si on reprend l'entraînement, utiliser le dernier LR fourni
    if resume_from and last_lr is not None:
        initial_lr = last_lr
        print(f"Using provided learning rate: {initial_lr:.14f}")
    # Ou si ultra_low_lr est demandé
    elif resume_from and ultra_low_lr:
        initial_lr = 5e-6  # Commencer très bas
        print(f"Using ultra-low starting learning rate: {initial_lr}")
        
    # Optimisation avec RAdam et scheduler ReduceLROnPlateau
    optimizer = optim.RAdam(net.parameters(), lr=initial_lr, weight_decay=2e-4)
    
    # Utilisation de ReduceLROnPlateau pour un ajustement adaptatif du learning rate
    # Configuration SANS limite inférieure pour le LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Surveiller la diminution de la perte
        factor=0.5,           # Facteur de réduction du LR (par 2)
        patience=5,           # Combien d'époques attendre avant de réduire le LR
        verbose=True,         # Afficher un message quand le LR est réduit
        min_lr=0,             # Pas de limite inférieure pour le LR
        cooldown=3            # Nombre d'époques à attendre après une réduction avant d'en faire une autre
    )
    
    # EarlyStopper avec grande patience pour éviter d'arrêter trop tôt
    stopper = EarlyStopper(patience=100)
    
    print(f"Training for {epochs} epochs starting from epoch {start_epoch}")
    
    # S'assurer que le fichier CSV existe avec les en-têtes appropriés
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            
            # Écrire les métriques existantes si disponibles
            if existing_metrics:
                for old_row in existing_metrics:
                    if int(old_row[0]) < start_epoch:
                        writer.writerow(old_row)
                print(f"Initialized metrics file with {len(existing_metrics)} previous records")
            else:
                print(f"Initialized empty metrics file")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        net.train()
        total_loss = 0
        
        # Libérer mémoire au début de l'époque
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + epochs - 1} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            loss = policy_loss_with_smoothing(logits, yb, label_smoothing)
            optimizer.zero_grad()
            
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            
        train_loss = total_loss / n_train

        # Appliquer le rafraîchissement du modèle périodiquement
        if epoch % 50 == 0:
            net, optimizer = refresh_model(net, optimizer, device)
        
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                loss = policy_loss_with_smoothing(logits, yb, label_smoothing)
                val_loss += loss.item() * xb.size(0)
        val_loss /= n_val
        
        # Calculer l'entropie moyenne des prédictions et des cibles
        with torch.no_grad():
            pred_entropies = []
            target_entropies = []
            batch_count = 0
            
            # Échantillonner quelques batches pour l'analyse
            for xb, yb in val_loader:
                if batch_count >= 5:  # Limiter à 5 batches pour l'efficacité
                    break
                    
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                probs = F.softmax(logits, dim=1)
                
                # Calculer l'entropie des prédictions
                log_probs = F.log_softmax(logits, dim=1)
                pred_entropy = -(probs * log_probs).sum(dim=1).mean().item() / np.log(2)  # Conversion en bits
                pred_entropies.append(pred_entropy)
                
                # Calculer l'entropie des cibles
                target_entropy = -(yb * torch.log2(yb + 1e-10)).sum(dim=1).mean().item()
                target_entropies.append(target_entropy)
                
                batch_count += 1
                
            if batch_count > 0:
                avg_pred_entropy = sum(pred_entropies) / len(pred_entropies)
                avg_target_entropy = sum(target_entropies) / len(target_entropies)
                print(f"  Entropie moyenne - Prédictions: {avg_pred_entropy:.4f} bits, Cibles: {avg_target_entropy:.4f} bits")

        # Libérer la mémoire GPU non utilisée après la validation
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Mise à jour du scheduler avec la perte de validation
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.14f}")

        # Ajuster min_delta en fonction du LR actuel
        stopper.adjust_for_lr(current_lr)
        
        # Enregistrer la ligne actuelle
        row = [
            epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.12f}", 
            f"{avg_pred_entropy:.4f}" if batch_count > 0 else "",
            f"{avg_target_entropy:.4f}" if batch_count > 0 else ""
        ]
        
        # Ajouter la nouvelle ligne au CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Créer un graphique avec toutes les données (anciennes + nouvelles)
        plot_config = [
            (['Train Loss', 'Val Loss'], None, 'Loss'),
            (['Pred Entropy', 'Target Entropy'], None, 'Entropy (bits)')
        ]
        
        # Lire toutes les données pour le plot
        all_data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for header in csv_headers:
                all_data[header] = []
            for line in reader:
                for header in csv_headers:
                    try:
                        if header in line and line[header] not in ["", None]:
                            try:
                                all_data[header].append(float(line[header]))
                            except (ValueError, TypeError):
                                # Skip invalid values
                                pass
                    except (ValueError, KeyError):
                        pass
        
        # Tracer les graphiques avec toutes les données
        plot_training_metrics(csv_path, output_dir, csv_headers)
        
        # Sauvegarder le meilleur modèle complet quand la validation s'améliore
        if val_loss <= stopper.best_loss - stopper.min_delta:
            best_model_path = os.path.join(output_dir, 'best_policynet_9x9.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }, best_model_path)
            
            # Également enregistrer l'information dans un fichier texte pour référence rapide
            best_info_path = os.path.join(output_dir, 'best_model_info.txt')
            with open(best_info_path, 'w') as f:
                f.write(f"Best model:\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Val Loss: {val_loss:.6f}\n")
                f.write(f"Train Loss: {train_loss:.6f}\n")
                f.write(f"Learning Rate: {current_lr:.14f}\n")
            
            print(f"New best model saved at epoch {epoch}")

        # Sauvegarder l'état complet à chaque époque (écrase la version précédente)
        last_model_path = os.path.join(output_dir, 'last_model_full.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        }, last_model_path)
        
    print("Training complete. Models saved.")


def resume_training(checkpoint_path, npz_path, epochs=1000, batch_size=512, 
                   label_smoothing=0.1):
    """
    Reprend l'entraînement à partir d'un checkpoint de manière fiable et simplifiée.
    
    Args:
        checkpoint_path: Chemin vers le modèle à charger
        npz_path: Chemin vers les données d'entraînement
        epochs: Nombre d'époques à exécuter
        batch_size: Taille des batchs
        label_smoothing: Paramètre de label smoothing
    """
    print(f"Resuming training from: {checkpoint_path}")
    
    # Charger le checkpoint
    try:
        checkpoint = torch.load(checkpoint_path)
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            raise ValueError("Le checkpoint ne contient pas les informations requises")
        
        # Extraire les informations nécessaires
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        epoch = checkpoint.get('epoch')
        learning_rate = checkpoint.get('learning_rate')
        
        if epoch is None or learning_rate is None:
            raise ValueError("Epoch ou learning rate manquant dans le checkpoint")
            
        print(f"Checkpoint chargé avec succès: époque {epoch}, LR {learning_rate:.14f}")
    except Exception as e:
        print(f"Erreur lors du chargement du checkpoint: {e}")
        print("Impossible de reprendre l'entraînement. Utilisez --fresh pour démarrer un nouvel entraînement.")
        return
    
    # Identifier le répertoire source des métriques
    source_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    csv_path = os.path.join(source_dir, 'metrics_all.csv')
    
    # Charger les métriques existantes
    existing_metrics = []
    if os.path.exists(csv_path):
        csv_headers, existing_metrics, _, _ = load_metrics_from_csv(csv_path)
        print(f"Métriques chargées: {len(existing_metrics)} époques")
    else:
        print("Aucune métrique existante trouvée")
    
    # Créer un nouveau répertoire pour cette reprise basé sur le dossier original
    source_folder_name = os.path.basename(source_dir)
    
    # Créer le nom du nouveau dossier avec _resumed à la fin
    new_folder_name = f"{source_folder_name}_resumed"
    
    # Si le dossier existe déjà, ajouter un index incrémental
    base_path = os.path.dirname(os.path.abspath(__file__))
    counter = 1
    new_output_dir = os.path.join(base_path, new_folder_name)
    
    while os.path.exists(new_output_dir):
        new_folder_name = f"{source_folder_name}_resumed_{counter}"
        new_output_dir = os.path.join(base_path, new_folder_name)
        counter += 1
    
    os.makedirs(new_output_dir, exist_ok=True)
    print(f"Résultats dans: {new_output_dir}")
    
    # Démarrer l'entraînement avec l'historique complet
    train_policy_9x9(
        npz_path=npz_path,
        batch_size=batch_size,
        epochs=epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        label_smoothing=label_smoothing,
        resume_from=checkpoint_path,
        continue_output_dir=new_output_dir,
        existing_metrics=existing_metrics,  # Passer les métriques existantes
        start_epoch=epoch + 1,  # Commencer à l'époque suivante
        last_lr=learning_rate
    )


# python3 go_policy_9x9_v5.py --resume training_logs_policy9x9_v5_2025-06-06_18-21-11/last_model_full.pth --epochs 10000 --bs 512 --label_smoothing 0.05

if __name__ == '__main__':
    import argparse
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.join('dataset', 'policy_dataset', 'policy_dataset_9x9.npz')
    default_data_path = os.path.join(base_dir, rel_path)
    
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=default_data_path, help='Path to training data')
    p.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train')
    p.add_argument('--bs', type=int, default=512, help='Batch size')
    p.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                  help='Device to use for training')
    p.add_argument('--label_smoothing', type=float, default=0.1, 
                  help='Amount of label smoothing to apply (0-1)')
    p.add_argument('--resume', type=str, default=None, 
                  help='Path to model checkpoint to resume training')
    p.add_argument('--fresh', action='store_true', 
                  help='Start a fresh training run (ignore previous metrics)')
    
    args = p.parse_args()
    
    if args.resume and not args.fresh:
        resume_training(
            checkpoint_path=args.resume,
            npz_path=args.data,
            epochs=args.epochs,
            batch_size=args.bs,
            label_smoothing=args.label_smoothing
        )
    else:
        train_policy_9x9(
            args.data,
            batch_size=args.bs,
            lr=args.lr,
            epochs=args.epochs, 
            device=args.device,
            label_smoothing=args.label_smoothing
        )

