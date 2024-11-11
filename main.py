import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter


class Tokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]
        self.word_counts = Counter()
        self.is_fitted = False

    def fit(self, texts):
        # Count all words
        for text in texts:
            words = self._preprocess(text)
            self.word_counts.update(words)

        # Get most common words
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)  # -2 for <pad> and <unk>

        # Build vocabulary
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

        self.is_fitted = True

    def _preprocess(self, text):
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def encode(self, text, max_length):
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")

        words = self._preprocess(text)
        # Convert words to indices
        indices = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]

        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.word2idx["<pad>"]] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]

        return indices

    def vocab_size(self):
        return len(self.word2idx)


class DynamicCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_classes, max_seq_length=200):
        super(DynamicCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(64 * (max_seq_length // 4), n_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        x = self.pool(torch.relu(self.conv1(embedded)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)

        x = x.flatten(1)
        x = self.fc(x)
        return x

    def update_output_layer(self, new_n_classes):
        old_fc = self.fc
        in_features = old_fc.in_features
        self.fc = nn.Linear(in_features, new_n_classes)

        min_classes = min(old_fc.out_features, new_n_classes)
        self.fc.weight.data[:min_classes] = old_fc.weight.data[:min_classes]
        self.fc.bias.data[:min_classes] = old_fc.bias.data[:min_classes]


class TextDataStream:
    def __init__(self, batch_size=32, max_seq_length=200):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        # Load dataset
        print("Loading dataset...")
        self.dataset = load_dataset("ag_news")

        # Initialize tokenizer
        print("Building vocabulary...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit([item['text'] for item in self.dataset['train']])

        self.current_index = 0
        self.label_encoder = LabelEncoder()

        # Initialize class tracking - Start with all classes active
        self.seen_classes = set(range(4))  # AG News has 4 classes
        self.class_counts = defaultdict(int)

        # Initialize label encoder with all possible classes
        self.label_encoder.fit(list(range(4)))

        # Metrics tracking
        self.current_accuracy = 0
        self.correct_predictions = 0
        self.total_predictions = 0

    def text_to_tensor(self, text):
        indices = self.tokenizer.encode(text, self.max_seq_length)
        return torch.tensor(indices)

    def update_metrics(self, predictions, labels):
        correct = (predictions == labels).sum().item()
        self.correct_predictions += correct
        self.total_predictions += len(labels)
        self.current_accuracy = (self.correct_predictions / self.total_predictions) * 100

    def get_next_batch(self):
        texts, labels = [], []

        while len(texts) < self.batch_size:
            if self.current_index >= len(self.dataset['train']):
                self.current_index = 0
                self.simulate_concept_drift()

            item = self.dataset['train'][self.current_index]
            self.current_index += 1

            if item['label'] in self.seen_classes:
                texts.append(self.text_to_tensor(item['text']))
                labels.append(item['label'])
                self.class_counts[item['label']] += 1

            if len(texts) >= self.batch_size:
                break

        if not texts:
            return None, None

        return (torch.stack(texts),
                torch.tensor(self.label_encoder.transform(labels)))

    def simulate_concept_drift(self):
        """Simulate concept drift with more conservative changes"""
        if random.random() < 0.05:  # Reduced probability of drift
            if random.random() < 0.5 and len(self.seen_classes) > 2:
                # Remove a class
                remove_class = random.choice(list(self.seen_classes))
                self.seen_classes.remove(remove_class)
                print(f"Removed class {remove_class}. Active classes: {self.seen_classes}")
            else:
                # Add a new class
                available_classes = set(range(4)) - self.seen_classes
                if available_classes:
                    new_class = random.choice(list(available_classes))
                    self.seen_classes.add(new_class)
                    print(f"Added class {new_class}. Active classes: {self.seen_classes}")
# Update only the LearningVisualizer class in your code:

class LearningVisualizer:
    def __init__(self):
        # Use a basic style instead of seaborn
        plt.style.use('default')

        # Set up the figure
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.patch.set_facecolor('white')  # Set white background
        self.fig.suptitle('Incremental Learning Metrics', fontsize=16)

        # Create subplots with grid
        self.loss_ax = self.fig.add_subplot(221)
        self.class_dist_ax = self.fig.add_subplot(222)
        self.accuracy_ax = self.fig.add_subplot(223)
        self.concept_drift_ax = self.fig.add_subplot(224)

        # Initialize data storage
        self.losses = []
        self.accuracies = []
        self.class_distributions = []
        self.class_changes = []
        self.batch_numbers = []

        # Define colors manually instead of using seaborn
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']

    def update_plots(self, batch_idx, loss, accuracy, class_counts, n_classes):
        self.batch_numbers.append(batch_idx)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.class_changes.append(n_classes)
        self.class_distributions.append(dict(class_counts))

        # Clear all axes
        for ax in [self.loss_ax, self.class_dist_ax, self.accuracy_ax, self.concept_drift_ax]:
            ax.clear()

        # Plot loss
        self.loss_ax.plot(self.batch_numbers, self.losses, 'b-', linewidth=2)
        self.loss_ax.set_title('Training Loss', fontsize=12)
        self.loss_ax.set_xlabel('Batch', fontsize=10)
        self.loss_ax.set_ylabel('Loss', fontsize=10)
        self.loss_ax.grid(True, linestyle='--', alpha=0.7)

        # Plot class distribution
        if self.class_distributions:
            latest_dist = self.class_distributions[-1]
            classes = list(latest_dist.keys())
            values = list(latest_dist.values())
            self.class_dist_ax.bar(classes, values,
                                   color=[self.colors[i % len(self.colors)] for i in range(len(classes))])
            self.class_dist_ax.set_title('Current Class Distribution', fontsize=12)
            self.class_dist_ax.set_xlabel('Class', fontsize=10)
            self.class_dist_ax.set_ylabel('Count', fontsize=10)
            self.class_dist_ax.grid(True, linestyle='--', alpha=0.7)

        # Plot accuracy
        self.accuracy_ax.plot(self.batch_numbers, self.accuracies, 'g-', linewidth=2)
        self.accuracy_ax.set_title('Classification Accuracy', fontsize=12)
        self.accuracy_ax.set_xlabel('Batch', fontsize=10)
        self.accuracy_ax.set_ylabel('Accuracy (%)', fontsize=10)
        self.accuracy_ax.grid(True, linestyle='--', alpha=0.7)

        # Plot concept drift
        self.concept_drift_ax.plot(self.batch_numbers, self.class_changes, 'r-', linewidth=2)
        self.concept_drift_ax.set_title('Number of Active Classes', fontsize=12)
        self.concept_drift_ax.set_xlabel('Batch', fontsize=10)
        self.concept_drift_ax.set_ylabel('Number of Classes', fontsize=10)
        self.concept_drift_ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout and display
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        try:
            plt.pause(0.1)
        except Exception as e:
            print(f"Warning: Display update failed: {e}")
            pass  # Continue even if display fails

def train_incremental():
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Initializing components...")
    stream = TextDataStream(batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    visualizer = LearningVisualizer()

    model = DynamicCNN(
        vocab_size=stream.tokenizer.vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        n_classes=4,
        max_seq_length=MAX_SEQ_LENGTH
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    n_batches = 1000
    running_loss = 0.0

    for batch_idx in range(n_batches):
        texts, labels = stream.get_next_batch()
        if texts is None:
            continue

        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        current_n_classes = len(stream.seen_classes)
        if current_n_classes != model.fc.out_features:
            model.update_output_layer(current_n_classes)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        stream.update_metrics(predictions, labels)
        running_loss += loss.item()

        if batch_idx % 10 == 9:
            avg_loss = running_loss / 10
            visualizer.update_plots(
                batch_idx,
                avg_loss,
                stream.current_accuracy,
                stream.class_counts,
                len(stream.seen_classes)
            )
            running_loss = 0.0

            print(f'[Batch {batch_idx + 1}] '
                  f'Loss: {avg_loss:.3f} '
                  f'Accuracy: {stream.current_accuracy:.2f}% '
                  f'Active Classes: {len(stream.seen_classes)}')


if __name__ == "__main__":
    plt.ion()
    print("Starting incremental learning...")
    train_incremental()
    plt.ioff()
    print("done")
    plt.show()
