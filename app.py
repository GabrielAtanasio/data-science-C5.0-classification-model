import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder # Para o caso da variável alvo ser textual

# --- 1. Carregar o conjunto de dados do arquivo CSV ---
file_path = 'all-cause.csv'  # AJUSTAR AQUI: Coloque o caminho correto se o arquivo não estiver na mesma pasta

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Verifique o caminho e o nome do arquivo.")
    exit()

print("--- Primeiras linhas do arquivo carregado: ---")
print(df.head())
print("\n--- Informações sobre as colunas: ---")
df.info()
print("\n")

# --- 2. Definir a variável alvo (y) e as características (X) ---
# AJUSTAR AQUI:
# Substitua 'NOME_DA_SUA_COLUNA_ALVO' pelo nome exato da coluna no seu CSV que você quer prever.
# Se a sua coluna alvo for textual (ex: 'Baixo', 'Médio', 'Alto'), precisaremos codificá-la.
target_column_name = 'Mortstat' # <--- MUITO IMPORTANTE AJUSTAR AQUI

if target_column_name not in df.columns:
    print(f"Erro: A coluna alvo '{target_column_name}' não foi encontrada no DataFrame.")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    exit()

X = df.drop(columns=[target_column_name]) # Usar todas as outras colunas como características
y = df[target_column_name]

# Se a sua variável alvo (y) for categórica (texto), ela precisa ser convertida para números.
# O DecisionTreeClassifier do scikit-learn lida bem com isso internamente para 'y',
# mas se precisar para 'X' (features), outras técnicas como OneHotEncoder seriam usadas.
# Vamos verificar se 'y' é numérica. Se não for, vamos usar LabelEncoder.
if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
    print(f"A coluna alvo '{target_column_name}' é categórica. Aplicando LabelEncoder...")
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Guardar as classes originais para referência
    target_classes = le.classes_
    print(f"Classes da variável alvo mapeadas para: {dict(zip(range(len(target_classes)), target_classes))}\n")
else:
    target_classes = sorted(y.unique()) # Para o caso de já ser numérica mas querermos saber as classes

# AJUSTAR AQUI (Opcional): Selecionar colunas específicas para X, se não quiser usar todas as outras
# Exemplo: feature_columns = ['coluna1', 'coluna3', 'coluna7']
# X = df[feature_columns]

# --- 3. Pré-processamento (Exemplo básico - pode precisar de mais) ---
# Verificar se há dados faltantes nas características X
if X.isnull().sum().any():
    print("Atenção: Existem dados faltantes nas características (X).")
    print("Considere tratá-los (ex: preenchimento com média/mediana, ou remoção).")
    # Exemplo simples de preenchimento com a média (apenas para colunas numéricas):
    for col in X.columns:
        if X[col].isnull().any() and pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
    print("Dados faltantes numéricos preenchidos com a média (verifique se é apropriado!).\n")

# Verificar se há colunas não numéricas em X (features)
non_numeric_cols_X = X.select_dtypes(include=['object', 'category']).columns
if not non_numeric_cols_X.empty:
    print(f"Atenção: As seguintes colunas de características (X) não são numéricas: {non_numeric_cols_X.tolist()}")
    print("Árvores de decisão no scikit-learn requerem input numérico.")
    print("Considere usar pd.get_dummies() para 'one-hot encoding' ou outras técnicas de codificação.")
    # Exemplo simples de one-hot encoding para colunas categóricas em X:
    # X = pd.get_dummies(X, columns=non_numeric_cols_X, drop_first=True)
    # print("Colunas categóricas em X foram transformadas com one-hot encoding (verifique!).\n")
    print("ERRO: Por favor, trate as colunas não numéricas em X antes de prosseguir.")
    exit()


# --- 4. Dividir os dados em conjuntos de treinamento e teste ---
# test_size=0.3 significa 30% para teste, 70% para treino.
# random_state garante que a divisão seja a mesma toda vez que o código rodar (reprodutibilidade).
# stratify=y é importante para problemas de classificação, para manter a proporção das classes
# nos conjuntos de treino e teste, especialmente se as classes forem desbalanceadas.
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
except ValueError as e:
    if "The least populated class in y has only 1 member, which is too few" in str(e):
        print("Erro ao dividir os dados: Uma ou mais classes na variável alvo têm poucos exemplos para permitir a estratificação.")
        print("Tente usar um test_size menor, coletar mais dados, ou não usar stratify=y (com cautela).")
        # Tentativa sem stratify (usar com cautela se as classes forem desbalanceadas)
        print("Tentando dividir sem estratificação...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        raise e


# --- 5. Criar e treinar o modelo de árvore de decisão (similar ao C5.0) ---
# Usamos 'entropy' como critério, relacionado ao ganho de informação (usado no C4.5/C5.0).
# 'max_depth' ajuda a evitar que a árvore cresça demais (overfitting).
# 'min_samples_leaf' define o número mínimo de amostras para ser uma folha.
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# --- 6. Fazer previsões no conjunto de teste ---
y_pred = model.predict(X_test)

# --- 7. Avaliar o modelo ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}\n")

print("--- Relatório de Classificação: ---")
# Se usamos LabelEncoder, precisamos dos nomes originais das classes para o relatório
if 'le' in locals(): # Verifica se LabelEncoder foi usado
    print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in target_classes]))
else:
    print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in target_classes]))


# --- 8. Visualizar as regras da árvore de decisão (opcional) ---
# O C5.0 é conhecido por gerar regras legíveis.
# Podemos exportar a árvore treinada como texto.
feature_names = X.columns.tolist() # Nomes das colunas usadas como características
class_names_str = [str(cls) for cls in target_classes] # Nomes das classes como strings

try:
    tree_rules = export_text(model, feature_names=feature_names, class_names=class_names_str)
    print("\n--- Regras da Árvore de Decisão (formato texto): ---")
    print(tree_rules)
except AttributeError as e:
    # Correção para versões mais antigas do scikit-learn que não têm class_names em export_text
    if 'class_names' in str(e):
        tree_rules = export_text(model, feature_names=feature_names)
        print("\n--- Regras da Árvore de Decisão (formato texto, sem nomes de classe explícitos nas folhas): ---")
        print(tree_rules)
        print(f"(Lembre-se do mapeamento de classes: {dict(zip(range(len(target_classes)), target_classes))})")
    else:
        raise e
except Exception as e:
    print(f"Não foi possível gerar as regras da árvore em formato texto: {e}")

# --- Exemplo de como visualizar a árvore (requer matplotlib e graphviz instalados) ---
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=feature_names, class_names=class_names_str, filled=True, rounded=True, fontsize=10)
plt.title("Visualização da Árvore de Decisão")
plt.show()
print("\nVisualização da árvore gerada (se matplotlib e graphviz estiverem configurados).")