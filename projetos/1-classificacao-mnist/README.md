# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Henrique Soares de Souza**

### 1️⃣ Resumo da Arquitetura do Modelo

Descreva, em palavras, a arquitetura da CNN implementada em `train_model.py` (número de blocos convolucionais, uso de batch normalization/dropout, estratégia de validação/early stopping).

R: No modelo criado, foram usados 3 blocos convolucionais, no formato descrito na orientação: Conv2D + BatchNormalization + MaxPooling2D.
Nesse caso, foram usadas as camadas de BatchNormalization, que serve principalmente para aceleração do treinamento, e redução de instabilidades durante o processo.
A estratégia de validação dos resultados foi usar uma parcela específica (validation_split(0.1), ou 10%) dos dados do conjunto mnist. Usou-se um dropout em uma camada Dense de neurônios, logo antes da camada final, com frequência de desativamento de 50% para regularização da rede. 
Foi usado o callback EarlyStopping com parâmetros apropriados para a aplicação: Monitora o valor do erro, ou "loss" ("val_loss"), tem paciência de 3 epochs (patience = 3), e, caso ativado, retorna para o epoch com os valores de peso mais efetivos (restore_best_weights = True).

### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

R: 

tensorflow 2.21.0
keras 3.15.0 (Nota: aparentemente o script do Github Actions não consegue instalar esta versão, logo o keras ficou sem versão especificada no requirements.txt)
numpy 2.4.6

absl-py            2.5.0
astunparse         1.6.3
certifi            2026.7.22
charset-normalizer 3.4.9
flatbuffers        25.12.19
gast               0.7.0
gitdb              4.0.12
GitPython          3.1.50
google-pasta       0.2.0
grpcio             1.83.0
h5py               3.14.0
idna               3.18
libclang           18.1.1
markdown-it-py     4.2.0
mdurl              0.1.2
ml_dtypes          0.5.4
namex              0.1.0
opt_einsum         3.4.0
optree             0.19.1
packaging          26.2
pip                26.1.2
protobuf           7.35.1
Pygments           2.20.0
requests           2.34.2
rich               15.0.0
setuptools         82.0.1
six                1.17.0
smmap              5.0.3
termcolor          3.3.0
typing_extensions  4.16.0
urllib3            2.7.0
wheel              0.46.3
wrapt              2.2.2

### 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo em `optimize_model.py`.

R: A técnica de otimização do modelo foi a Dynamic Range Quantization, juntamente com Weight Quantization,
ambas otimizações automaticamente aplicadas por meio de "converter.optimizations = [tf.lite.Optimize.DEFAULT]"

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

R: 
A acurácia obtida durante o treinamento do modelo foi de 0.9915000200271606, enquanto o tamanho dos arquivos "model.h5" e "model.tflite" foram 654 KB e 60 KB, respectivamente.

Ou seja, o arquivo otimizado tem 9,1% do tamanho original, aproximadamente. 


### 5️⃣ Comentários Adicionais (Opcional)

Dificuldades encontradas, decisões técnicas importantes, limitações do modelo, aprendizados durante o desafio.

R: A maior dificuldade do desafio, por uma gigante margem, foi a validação automatizada do projeto por meio dos scripts do GitHub Actions (que, infelizmente, não consegui corrigir). Apesar do modelo estar presente localmente, e funcionar para diversos elementos do conjunto de dados usado, houveram erros no workflow decorrentes principalmente do keras, por o mesmo estar agora separado do tensorflow (Keras 3), além do formato .h5 estar muito depreciado para a versão dos pacotes usados. 

O modelo é limitado em seu uso, principalmente por ser não tão profundo; ainda assim, demorou um tempo significativo para ser treinado.

O maior aprendizado durante o desafio foi em relação às diversas camadas usadas para uma CNN, seu modo de operação, e os conceitos envolvidos no software.

### 6️⃣ Exemplo de Inferência

Cole a saída do terminal ao rodar `run_inference.py` (predito vs. real para as 5+ amostras), e comente brevemente se houve algum caso interessante (acerto ou erro) entre as amostras testadas.

R: 
A inferência foi rodada para N=15 elementos:

Rodando inferencia em 15 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
Amostra 6: predito=1 | real=1
Amostra 7: predito=4 | real=4
Amostra 8: predito=9 | real=9
Amostra 9: predito=5 | real=5
Amostra 10: predito=9 | real=9
Amostra 11: predito=0 | real=0
Amostra 12: predito=6 | real=6
Amostra 13: predito=9 | real=9
Amostra 14: predito=0 | real=0
Amostra 15: predito=1 | real=1

Não houveram casos que se destacaram dessas amostras de dados. Todos apresetaram os resultados esperados da inferência, dado a acurácia elevada do modelo para essa aplicação.

