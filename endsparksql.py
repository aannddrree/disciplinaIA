# pyspark-shell
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover, HashingTF, IDF,
                                StringIndexer, OneHotEncoder, VectorAssembler)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (SparkSession.builder
         .appName("EnderecoScore-Logistica")
         .getOrCreate())

# =========================
# 1) Carregar CSV como tabela e inspecionar
# =========================
path = "enderecos_com_score.csv"  # ajuste caminho se necessário
(df := spark.read.option("header", True).option("inferSchema", True).csv(path)).createOrReplaceTempView("enderecos")

spark.sql("SELECT * FROM enderecos LIMIT 5").show(truncate=False)

# Checagem básica de colunas
expected = {"endereco_completo","cidade","estado","lat","lon","score"}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Colunas faltantes no CSV: {missing}")

# =========================
# 2) Criar alvo binário via quantil com Spark SQL
#    - usa percentile_approx(score, q)
#    - tenta sequência de quantis até garantir 2 classes
# =========================
quantis = [0.70, 0.75, 0.80, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

def rotular_por_quantil(q):
    spark.sql("DROP VIEW IF EXISTS thr_tbl")
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW thr_tbl AS
        SELECT percentile_approx(score, {q}) AS thr FROM enderecos
    """)
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW end_rot AS
        SELECT e.*,
               CASE WHEN e.score >= t.thr THEN 1 ELSE 0 END AS y
        FROM enderecos e CROSS JOIN thr_tbl t
    """)
    dist = spark.sql("SELECT y, COUNT(*) AS n FROM end_rot GROUP BY y").collect()
    return dist

used_q, thr_val = None, None
for q in quantis:
    dist = rotular_por_quantil(q)
    if len(dist) == 2:  # tem 0 e 1
        used_q = q
        thr_val = spark.sql("SELECT thr FROM thr_tbl").first()["thr"]
        break

if used_q is None:
    # fallback: mediana
    rotular_por_quantil(0.5)
    used_q = "mediana"
    thr_val = spark.sql("SELECT thr FROM thr_tbl").first()["thr"]

print(f"[INFO] Limiar usado: {thr_val:.2f} (quantil={used_q})")
spark.sql("SELECT y, COUNT(*) AS n FROM end_rot GROUP BY y").show()

# =========================
# 3) Split de treino/teste por cidade (evitar vazamento geográfico)
#    - usa um hash da cidade para dividir de forma estável
# =========================
spark.sql("""
CREATE OR REPLACE TEMP VIEW end_splitted AS
SELECT *,
       CASE WHEN pmod(abs(hash(cidade)), 10) < 7 THEN 'train' ELSE 'test' END AS split
FROM end_rot
""")

train_df = spark.sql("SELECT * FROM end_splitted WHERE split='train'").cache()
test_df  = spark.sql("SELECT * FROM end_splitted WHERE split='test'").cache()

print("[INFO] Train size:", train_df.count(), "| Test size:", test_df.count())

# =========================
# 4) Pipeline de features (texto + categóricos + numéricos)
# =========================
# Texto: endereco_completo -> tokens -> remove stopwords -> HashingTF -> IDF
tokenizer = RegexTokenizer(inputCol="endereco_completo", outputCol="tokens", pattern="\\W+")
stopper   = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
tf        = HashingTF(inputCol="tokens_clean", outputCol="tf", numFeatures=1<<18)
idf       = IDF(inputCol="tf", outputCol="text_tfidf")

# Categóricos: cidade, estado
idx_cidade = StringIndexer(inputCol="cidade", outputCol="cidade_idx", handleInvalid="keep")
idx_estado = StringIndexer(inputCol="estado", outputCol="estado_idx", handleInvalid="keep")
ohe        = OneHotEncoder(inputCols=["cidade_idx","estado_idx"],
                           outputCols=["cidade_ohe","estado_ohe"],
                           handleInvalid="keep")

# Numéricos
# (Se quiser padronizar, adicione StandardScaler; para simplicidade, mantemos como está.)
numeric_cols = ["lat","lon"]

# Montagem do vetor final
assembler = VectorAssembler(
    inputCols=["text_tfidf", "cidade_ohe", "estado_ohe"] + numeric_cols,
    outputCol="features",
    handleInvalid="keep"
)

# =========================
# 5) Treino: Regressão Logística (binária)
# =========================
lr = LogisticRegression(
    featuresCol="features",
    labelCol="y",
    maxIter=100,
    regParam=0.0,
    elasticNetParam=0.0,
    family="binomial"
)

pipeline = Pipeline(stages=[
    tokenizer, stopper, tf, idf,
    idx_cidade, idx_estado, ohe,
    assembler, lr
])

model = pipeline.fit(train_df)

# =========================
# 6) Avaliação (AUC-ROC e AUC-PR) no teste
# =========================
pred = model.transform(test_df)

e_roc = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction", labelCol="y", metricName="areaUnderROC"
)
e_pr  = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction", labelCol="y", metricName="areaUnderPR"
)

print("[TEST] AUC-ROC:", e_roc.evaluate(pred))
print("[TEST] AUC-PR :", e_pr.evaluate(pred))

# =========================
# 7) Ranqueamento de candidatos por probabilidade de "bom"
# =========================
candidatos_sql = """
SELECT stack(3,
 'Av. Paulista, 1000, Bela Vista','Sao Paulo','SP',-23.5614,-46.6559,
 'Rua Exemplo, 123, Centro',      'Sao Paulo','SP',-23.5440,-46.6330,
 'Av. Atlântica, 3000, Copacabana','Rio de Janeiro','RJ',-22.9717,-43.1830
) AS (endereco_completo, cidade, estado, lat, lon)
"""
spark.sql(candidatos_sql).createOrReplaceTempView("candidatos")

cand_pred = model.transform(spark.table("candidatos")) \
                 .select("endereco_completo","cidade","estado","lat","lon","probability","prediction") \
                 .withColumn("prob_bom", expr("probability[1]")) \
                 .drop("probability") \
                 .orderBy(col("prob_bom").desc())

cand_pred.show(truncate=False)

# =========================
# 8) (Opcional) Salvar modelo
# =========================
# model.write().overwrite().save("models/modelo_logistico_endereco_spark")
