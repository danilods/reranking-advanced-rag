# 🚗🔍 Busca Semântica e Re-Ranking de Carros com ChromaDB

Este repositório demonstra como usar ChromaDB para realizar buscas semânticas e re-ranking de descrições de carros utilizando embeddings gerados por modelos de linguagem.

## 📋 Descrição do Projeto

Este projeto visa melhorar a precisão e a relevância dos resultados de buscas semânticas utilizando técnicas avançadas de Recuperação de Informação Aumentada (RAG). Utilizamos o ChromaDB para indexar e buscar embeddings das descrições dos carros, e aplicamos re-ranking com um modelo Cross-Encoder.

## 📂 Estrutura do Projeto

- **`car_data_pt.csv`**: Arquivo CSV contendo os dados dos carros.
- **`main.py`**: Código principal para carregar dados, gerar embeddings, buscar e reordenar resultados.
- **`plot_embeddings.py`**: Função para plotar embeddings e visualizações.

## 🔧 Funcionalidades

1. **Carregamento dos Dados**: Carrega os dados dos carros a partir de um CSV.
2. **Geração de Embeddings**: Usa `SentenceTransformer` para gerar embeddings das descrições dos carros.
3. **Configuração do ChromaDB**: Indexa os embeddings no ChromaDB.
4. **Busca Semântica**: Realiza buscas semânticas utilizando ChromaDB.
5. **Re-Ranking**: Aplica re-ranking dos resultados utilizando um modelo Cross-Encoder.
6. **Visualização**: Gera visualizações dos embeddings antes e depois do re-ranking.

## 🗂️ Estrutura do Código

### `main.py`

```python
# Código principal para carregar dados, gerar embeddings, buscar e reordenar resultados
```

### `plot_embeddings.py`

```python
# Função para plotar embeddings e visualizações
```

## 🚀 Como Executar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/nome-do-repositorio.git
    cd nome-do-repositorio
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Execute o código:
    ```bash
    python main.py
    ```

## 📊 Visualizações

### Todos os Carros e Consulta
![Todos os Carros e Consulta](todos_carros_consulta.png)

### Antes do Re-Ranking
![Antes do Re-Ranking](antes_re_ranking.png)

### Após o Re-Ranking
![Após o Re-Ranking](apos_re_ranking.png)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Feito com ❤️ por [Seu Nome](https://github.com/seu-usuario).

---

### Referências

- [ChromaDB](https://chromadb.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [Cross-Encoder](https://www.sbert.net/docs/pretrained_cross-encoders.html)
