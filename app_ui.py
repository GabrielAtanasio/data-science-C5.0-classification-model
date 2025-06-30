import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Árvore de Decisão", layout="wide")
st.title("Ferramenta Interativa de Análise com Árvore de Decisão 🌳")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Configurações")
    
    # 1. Upload do arquivo
    uploaded_file = st.file_uploader("1. Carregue seu arquivo CSV", type=["csv"])
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

if df is not None:
    # Mostra uma prévia dos dados na área principal
    st.header("Pré-visualização dos Dados")
    st.dataframe(df.head())

    # Buffer para capturar o output do df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    with st.expander("Ver informações detalhadas do DataFrame"):
        st.text(info_str)

    with st.sidebar:
        # 2. Seleção de colunas
        st.subheader("2. Definição de Variáveis")
        available_columns = df.columns.tolist()
        target_column = st.selectbox("Selecione a Coluna Alvo (Y)", options=available_columns)
        
        # As features são todas as outras colunas por padrão
        default_features = [col for col in available_columns if col != target_column]
        feature_columns = st.multiselect("Selecione as Colunas de Características (X)", 
                                         options=available_columns, 
                                         default=default_features)

        # 3. Parâmetros do modelo
        st.subheader("3. Parâmetros do Modelo")
        max_depth = st.slider("Profundidade Máxima da Árvore", min_value=2, max_value=30, value=5, step=1)
        min_samples_leaf = st.slider("Mínimo de Amostras por Folha", min_value=1, max_value=100, value=5, step=1)
        test_size = st.slider("Tamanho do Conjunto de Teste (%)", min_value=10, max_value=50, value=30, step=5) / 100.0

        # Botão para iniciar o treinamento
        if st.button("Treinar Modelo e Analisar", type="primary"):
            # --- Início do Processo de ML (adaptado do seu script) ---
            
            # Validação
            if not target_column or not feature_columns:
                st.error("Por favor, selecione a coluna alvo e pelo menos uma coluna de característica.")
            else:
                with st.spinner("Processando... Por favor, aguarde."):
                    try:
                        # Preparação dos dados
                        X = df[feature_columns]
                        y = df[target_column]
                        
                        # Tratamento de colunas categóricas em X (exemplo com one-hot encoding)
                        X = pd.get_dummies(X, drop_first=True)
                        feature_names = X.columns.tolist()

                        # Label Encoding para a variável alvo, se necessário
                        le = LabelEncoder()
                        if y.dtype == 'object':
                            y = le.fit_transform(y)
                            target_classes = le.classes_
                        else:
                            target_classes = sorted(y.unique())
                        
                        # Divisão treino/teste
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                        
                        # Treinamento do modelo
                        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Previsões e Avaliação
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, target_names=[str(cls) for cls in target_classes])
                        
                        # Regras da árvore
                        tree_rules = export_text(model, feature_names=feature_names, class_names=[str(cls) for cls in target_classes])
                        
                        # --- Apresentação dos Resultados ---
                        st.header("📊 Resultados da Análise")
                        
                        st.metric(label="Acurácia do Modelo no Conjunto de Teste", value=f"{accuracy:.4f}")
                        
                        st.subheader("Relatório de Classificação")
                        st.text(report)
                        
                        # Abas para regras e visualização
                        tab1, tab2 = st.tabs(["📜 Regras da Árvore (Texto)", "🖼️ Visualização Gráfica"])
                        
                        with tab1:
                            st.code(tree_rules, language='bash')
                        
                        with tab2:
                            fig, ax = plt.subplots(figsize=(25, 15))
                            plot_tree(model, 
                                      feature_names=feature_names, 
                                      class_names=[str(cls) for cls in target_classes], 
                                      filled=True, 
                                      rounded=True, 
                                      fontsize=10)
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a análise: {e}")
else:
    st.info("Aguardando o upload de um arquivo CSV...")