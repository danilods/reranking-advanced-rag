# Importar bibliotecas necessárias
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import chromadb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Normalizar vetores para similaridade de cosseno
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Definir a função para plotar embeddings
def plot_embeddings(embeddings, car_data, query_embedding, query_label, title, filename):
    # Reduzir dimensionalidade para 2D com PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Reduzir a dimensionalidade do embedding da consulta
    query_reduced = pca.transform(query_embedding.reshape(1, -1))

    # Plotar os embeddings dos carros
    plt.figure(figsize=(12, 12))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', label='Carros')
    
    # Adicionar anotações aos pontos
    for i, row in car_data.iterrows():
        plt.annotate(f"{row['modelo']}\n{row['potencia']} HP, R${row['preco']}, {row['ano']}", 
                     (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    # Plotar o embedding da consulta
    plt.scatter(query_reduced[:, 0], query_reduced[:, 1], c='red', label='Consulta', marker='x')
    plt.annotate(query_label, (query_reduced[0, 0], query_reduced[0, 1]), color='red')

    # Configurar o título e a legenda
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Definir a função principal
def main():
    # Carregar dados reais de carros a partir do arquivo CSV
    car_df = pd.read_csv("car_data_pt.csv")  # Assegure-se de salvar o CSV com o nome "car_data_pt.csv"

    # Debug: Mostrar os dados dos carros carregados
    print("Dados dos carros carregados:")
    print(car_df.head())

    # Consulta do cliente
    client_query = "carros esportivos com alta performance e baixo custo"

    # Preparar descrições dos carros
    car_descriptions = car_df.apply(
        lambda x: f"{x['ano']} {x['fabricante']} {x['modelo']} com {x['potencia']} HP, {x['cavalos_rodas']} cilindros, "
                  f"rodas motrizes {x['rodas_motrizes']}, consumo na estrada de {x['consumo_estrada']} MPG, preço R${x['preco']}",
        axis=1
    ).tolist()

    # Debug: Mostrar descrições dos carros
    print("Descrições dos carros:")
    for desc in car_descriptions:
        print(desc)

    # Preparar embeddings
    bi_encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
    car_embeddings = bi_encoder_model.encode(car_descriptions)

    # Normalizar embeddings para similaridade de cosseno
    car_embeddings = normalize(car_embeddings)

    # Configurar o ChromaDB
    client = chromadb.Client()

    # Limpar o índice ao deletar a coleção existente
    if "car_collection" in [col['name'] for col in client.list_collections()]:
        client.delete_collection(name="car_collection")
        
    collection = client.create_collection(name="car_collection")

    # Adicionar embeddings ao ChromaDB
    for i, embedding in enumerate(car_embeddings):
        collection.add(ids=[str(i)], embeddings=[embedding.tolist()])

    # Transformar a consulta em embedding
    query_embedding = bi_encoder_model.encode(client_query)
    query_embedding = normalize(query_embedding.reshape(1, -1))

    # Debug: Mostrar embedding da consulta
    print("Embedding da consulta:")
    print(query_embedding)

    # Busca semântica para recuperação inicial usando ChromaDB
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5, similarity_metric="cosine")
    initial_results_indices = [int(res) for res in results['ids'][0]]
    initial_results = car_df.iloc[initial_results_indices].reset_index(drop=True)
    initial_results_descriptions = [car_descriptions[idx] for idx in initial_results_indices]
    initial_scores = results['distances'][0]
    initial_results_embeddings = car_embeddings[initial_results_indices]

    # Debug: Mostrar resultados iniciais da busca
    print("Resultados iniciais da busca (descrições e pontuações):")
    for desc, score in zip(initial_results_descriptions, initial_scores):
        print(f"{desc} - Pontuação: {score}")

    # Criar tabela de resultados iniciais
    initial_results['score'] = initial_scores
    print("Tabela de resultados iniciais:")
    print(initial_results[['fabricante', 'modelo', 'ano', 'potencia', 'cavalos_rodas', 'preco', 'score']])

    # Preparar pares para o Cross-Encoder
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
    cross_encoder_inputs = [(client_query, desc) for desc in initial_results_descriptions]
    cross_encoder_scores = cross_encoder_model.predict(cross_encoder_inputs)

    # Debug: Mostrar pontuações do Cross-Encoder
    print("Pontuações do Cross-Encoder:")
    print(cross_encoder_scores)

    # Re-Ranking com Cross-Encoder
    initial_results['relevance'] = cross_encoder_scores
    reordered_results = initial_results.sort_values(by='relevance', ascending=False).reset_index(drop=True)

    # Debug: Mostrar descrições e relevâncias após o re-ranking
    print("Descrições e relevâncias após o re-ranking:")
    print(reordered_results[['modelo', 'relevance']])

    # Criar tabela de resultados após o re-ranking
    print("Tabela de resultados após o re-ranking:")
    print(reordered_results[['fabricante', 'modelo', 'ano', 'potencia', 'cavalos_rodas', 'preco', 'relevance']])

    # Debug: Mostrar os índices reordenados
    reordered_indices = [car_df[car_df['modelo'] == model].index[0] for model in reordered_results['modelo']]
    print("Índices reordenados:")
    print(reordered_indices)

    reordered_embeddings = car_embeddings[reordered_indices]

    # Visualização dos Embeddings com a Consulta
    plot_embeddings(np.vstack([car_embeddings, query_embedding]), car_df, query_embedding, client_query, 'Todos os Carros e Consulta', 'todos_carros_consulta.png')
    plot_embeddings(initial_results_embeddings, initial_results, query_embedding, client_query, 'Antes do Re-Ranking', 'antes_re_ranking.png')
    plot_embeddings(reordered_embeddings, reordered_results, query_embedding, client_query, 'Após o Re-Ranking', 'apos_re_ranking.png')

# Executar a função principal
if __name__ == "__main__":
    main()
