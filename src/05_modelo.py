# 1. Instalación (por si reiniciaste sesión)
!pip install transformers[torch] datasets scikit-learn tqdm -q

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
PATH_GS = "5_gold_standard_final.csv"
COMPONENTES_IIA = ["AR", "MO", "DI", "PO", "NA", "GO"]
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 80
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5

# ==========================================
# 3. DATASET (Cambio: Llamada directa al tokenizer)
# ==========================================
class NarcoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        # Usamos __call__ en lugar de encode_plus para evitar el AttributeError
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# ==========================================
# 4. MODELO (Cambio: Uso del token CLS directo)
# ==========================================
class MultiHeadNarcoClassifier(nn.Module):
    def __init__(self):
        super(MultiHeadNarcoClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.heads = nn.ModuleDict({
            comp: nn.Linear(self.bert.config.hidden_size, 4) for comp in COMPONENTES_IIA
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # En lugar de pooled_output, usamos el CLS token (índice 0)
        # Esto soluciona el problema de los 'MISSING pooler weights'
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        output = self.drop(cls_output)
        return {comp: self.heads[comp](output) for comp in COMPONENTES_IIA}

# ==========================================
# 5. FUNCIÓN DE ENTRENAMIENTO
# ==========================================
def entrenar_cerebro():
    if not os.path.exists(PATH_GS):
        print(f"❌ ERROR: Sube '{PATH_GS}' al panel izquierdo.")
        return

    df = pd.read_csv(PATH_GS)
    label_cols = [f"val_{c}" for c in COMPONENTES_IIA]
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(NarcoDataset(df_train.verso_texto.values, df_train[label_cols].values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NarcoDataset(df_val.verso_texto.values, df_val[label_cols].values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    model = MultiHeadNarcoClassifier().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"🔥 Entrenando en {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            total_loss = sum(loss_fn(outputs[comp], labels[:, i]) for i, comp in enumerate(COMPONENTES_IIA))

            train_losses.append(total_loss.item())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                v_loss = sum(loss_fn(outputs[comp], labels[:, i]) for i, comp in enumerate(COMPONENTES_IIA))
                val_losses.append(v_loss.item())

        print(f"✅ Epoch {epoch+1} - Loss Train: {np.mean(train_losses):.4f} | Loss Val: {np.mean(val_losses):.4f}")

    torch.save(model.state_dict(), "modelo_narco_hexadimensional.bin")
    print("\n🏆 ENTRENAMIENTO FINALIZADO. Modelo guardado.")

if __name__ == "__main__":
    entrenar_cerebro()