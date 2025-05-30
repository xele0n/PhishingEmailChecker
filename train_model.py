import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub
import os
import glob
import multiprocessing
import time
import datetime
import json

class PhishingEmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PhishingEmailClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(PhishingEmailClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_device_info():
    """Get information about available devices"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"CUDA is available with {num_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
        return True, num_gpus
    else:
        print("CUDA is not available. Using CPU.")
        return False, 0

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

def save_checkpoint(model, optimizer, scaler, epoch, batch_idx, loss, best_val_acc, checkpoint_dir='checkpoints'):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle DataParallel case
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'best_val_acc': best_val_acc,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scaler=None):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state (handle DataParallel case)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"✅ Checkpoint loaded - Epoch: {checkpoint['epoch']}, Batch: {checkpoint['batch_idx']}")
    return checkpoint

def validate_model(model, val_loader, criterion, device):
    """Validate the model and return metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def print_validation_report(val_loss, val_accuracy, all_predictions, all_labels, epoch):
    """Print detailed validation report"""
    print(f"\n📊 VALIDATION RESULTS - Epoch {epoch}")
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['Legitimate', 'Phishing'], 
                                 output_dict=True)
    
    print(f"   Precision (Legitimate): {report['Legitimate']['precision']:.4f}")
    print(f"   Recall (Legitimate): {report['Legitimate']['recall']:.4f}")
    print(f"   F1-Score (Legitimate): {report['Legitimate']['f1-score']:.4f}")
    print(f"   Precision (Phishing): {report['Phishing']['precision']:.4f}")
    print(f"   Recall (Phishing): {report['Phishing']['recall']:.4f}")
    print(f"   F1-Score (Phishing): {report['Phishing']['f1-score']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"   Confusion Matrix:")
    print(f"     Predicted: [Legit, Phish]")
    print(f"     Legit:     [{cm[0][0]:>5}, {cm[0][1]:>5}]")
    print(f"     Phish:     [{cm[1][0]:>5}, {cm[1][1]:>5}]")
    print()

def train_model(resume_from_checkpoint=None):
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Find the CSV file in the dataset directory
    csv_files = glob.glob(os.path.join(dataset_path, "**/*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the dataset directory")
    
    csv_file = csv_files[0]
    print(f"Found CSV file: {csv_file}")
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(csv_file)
    # Combine subject and body into a single text column
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['text'] = df['subject'] + ' ' + df['body']
    # Use label as target, ensure it's int
    df['label'] = df['label'].astype(int)
    print(f"Dataset loaded with {len(df)} samples")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    print("Initializing model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = PhishingEmailClassifier()
    
    # Device setup
    cuda_available, num_gpus = get_device_info()
    
    if cuda_available:
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # Use DataParallel for multi-GPU training
        if num_gpus > 1:
            print(f"Using DataParallel with {num_gpus} GPUs")
            model = nn.DataParallel(model)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PhishingEmailDataset(train_texts, train_labels, tokenizer)
    val_dataset = PhishingEmailDataset(val_texts, val_labels, tokenizer)
    
    # Determine optimal number of workers for data loading (use CPU cores)
    num_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overhead
    print(f"Using {num_workers} CPU workers for data loading")
    
    # Adjust batch size based on available resources
    if cuda_available:
        # Larger batch size for GPU
        batch_size = 16 if num_gpus == 1 else 8 * num_gpus
    else:
        # Smaller batch size for CPU
        batch_size = 4
    
    print(f"Using batch size: {batch_size}")
    
    # Create dataloaders with CPU workers for data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cuda_available,  # Pin memory for faster GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=cuda_available,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training for GPU (if available)
    scaler = None
    if cuda_available:
        scaler = torch.amp.GradScaler('cuda')
        print("Using mixed precision training")
    
    # Training parameters
    num_epochs = 3
    start_epoch = 0
    start_batch = 0
    best_val_acc = 0.0
    
    # Load checkpoint if specified
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'latest':
            checkpoint_path = 'checkpoints/latest_checkpoint.pth'
        else:
            checkpoint_path = resume_from_checkpoint
            
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch_idx'] + 1  # Resume from next batch
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"🔄 Resuming training from epoch {start_epoch}, batch {start_batch}")
    
    # Training loop
    print("Starting training...")
    total_batches = len(train_loader) * num_epochs
    total_samples = len(train_dataset) * num_epochs
    
    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time
    overall_batch_count = start_epoch * len(train_loader) + start_batch
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0
            epoch_start_time = time.time()
            
            # Skip batches if resuming mid-epoch
            skip_batches = start_batch if epoch == start_epoch else 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Skip batches if resuming
                if batch_idx < skip_batches:
                    continue
                    
                batch_start_time = time.time()
                
                # Move data to device (non_blocking for better GPU utilization)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if cuda_available:
                    # Use mixed precision for GPU training
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training for CPU
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                overall_batch_count += 1
                
                # Save checkpoint every 500 batches
                if (batch_idx + 1) % 500 == 0:
                    save_checkpoint(model, optimizer, scaler, epoch, batch_idx, loss.item(), best_val_acc)
                
                # Progress display every 30 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 30.0:
                    elapsed_time = current_time - start_time
                    samples_processed = overall_batch_count * batch_size
                    samples_remaining = total_samples - samples_processed
                    batches_remaining = total_batches - overall_batch_count
                    
                    # Calculate ETA
                    if overall_batch_count > 0:
                        avg_time_per_batch = elapsed_time / overall_batch_count
                        eta_seconds = avg_time_per_batch * batches_remaining
                        eta_str = format_time(eta_seconds)
                    else:
                        eta_str = "Calculating..."
                    
                    progress_percent = (overall_batch_count / total_batches) * 100
                    
                    print(f"\n📊 PROGRESS UPDATE ({datetime.datetime.now().strftime('%H:%M:%S')})")
                    print(f"   Epoch: {epoch + 1}/{num_epochs} | Batch: {batch_idx + 1}/{len(train_loader)}")
                    print(f"   Overall Progress: {progress_percent:.1f}% ({overall_batch_count}/{total_batches} batches)")
                    print(f"   Samples Processed: {samples_processed:,} / {total_samples:,}")
                    print(f"   Batches Remaining: {batches_remaining:,}")
                    print(f"   Current Loss: {loss.item():.4f}")
                    print(f"   Elapsed Time: {format_time(elapsed_time)}")
                    print(f"   ETA: {eta_str}")
                    print(f"   Speed: {avg_time_per_batch:.2f}s/batch\n" if overall_batch_count > 0 else "   Speed: Calculating...\n")
                    
                    last_progress_time = current_time
                
                # Print progress every 100 batches (in addition to time-based updates)
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Reset start_batch for next epoch
            start_batch = 0
            
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            print(f'✅ Epoch {epoch + 1}/{num_epochs} Complete - Average Loss: {avg_loss:.4f} - Time: {format_time(epoch_time)}')
            
            # Validation after each epoch
            print(f"\n🧪 Running validation...")
            val_start_time = time.time()
            val_loss, val_accuracy, all_predictions, all_labels = validate_model(model, val_loader, criterion, device)
            val_time = time.time() - val_start_time
            
            print_validation_report(val_loss, val_accuracy, all_predictions, all_labels, epoch + 1)
            print(f"   Validation Time: {format_time(val_time)}")
            
            # Save checkpoint after validation
            is_best = val_accuracy > best_val_acc
            if is_best:
                best_val_acc = val_accuracy
                print(f"🏆 New best validation accuracy: {best_val_acc:.4f}")
                
                # Save best model
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(model_state, 'best_phishing_model.pth')
                print("💎 Best model saved!")
            
            # Save regular checkpoint
            save_checkpoint(model, optimizer, scaler, epoch, len(train_loader)-1, avg_loss, best_val_acc)
            
            # Early stopping check (optional)
            if val_accuracy > 0.95:  # Stop if we achieve 95% accuracy
                print(f"🎯 Early stopping triggered - Validation accuracy reached {val_accuracy:.4f}")
                break
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user!")
        # Save checkpoint when interrupted
        save_checkpoint(model, optimizer, scaler, epoch, batch_idx, loss.item(), best_val_acc)
        print("💾 Progress saved. You can resume training later.")
        return
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training Complete!")
    print(f"   Total Time: {format_time(total_time)}")
    print(f"   Total Batches Processed: {overall_batch_count:,}")
    print(f"   Total Samples Processed: {overall_batch_count * batch_size:,}")
    print(f"   Average Speed: {total_time / overall_batch_count:.2f}s/batch")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Save the final model
    print("Saving final model...")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), 'phishing_model.pth')
    else:
        torch.save(model.state_dict(), 'phishing_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to resume from checkpoint
    if len(sys.argv) > 1:
        if sys.argv[1] == '--resume-latest':
            train_model(resume_from_checkpoint='latest')
        elif sys.argv[1] == '--resume':
            if len(sys.argv) > 2:
                train_model(resume_from_checkpoint=sys.argv[2])
            else:
                print("Please specify checkpoint path: python train_model.py --resume path/to/checkpoint.pth")
        else:
            print("Usage:")
            print("  python train_model.py                    # Start new training")
            print("  python train_model.py --resume-latest    # Resume from latest checkpoint")
            print("  python train_model.py --resume <path>    # Resume from specific checkpoint")
    else:
        train_model() 