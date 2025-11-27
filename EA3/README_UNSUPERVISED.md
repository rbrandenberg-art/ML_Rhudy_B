# Comprencion del negocio y entendimiento de los datos
El objetivo de este proyecto es construir un modelo de scoring crediticio  que prediga la probabilidad de incumplimiento o mora de un cliente.

En primer lugar se escogio utilizar las tablas application_.parquet(Tabla principal), previous_application.parquet(historial de anteriores prestamos), bureau.parquet(historial de crediticio de clientes), bureau_balance.parquet(historial mes a mes del estado de los creditos). No se usaron las otras tablas, pues una es de informacion y las otras podian llenar con demaciados datos no tan relevantes.
Luego se procede a unir las tablas para trabajar en una vista completa los riesgos de los clientes en una tabla nueva llamada df_final_credit_scoring_merged(tablas elegidas unidas)

Riesgo Integral: La información relevante para el riesgo crediticio no se limita a la solicitud actual (application_.parquet); se encuentra dispersa en el historial con el buró de crédito (bureau.parquet, bureau_balance.parquet) y las interacciones previas con la institución (previous_application.parquet).

Requerimiento del Modelo: La mayoría de los modelos de Machine Learning (incluidos los que se usarán para scoring y la técnica no supervisada) requieren que los datos de entrada estén en una sola estructura tabular, donde cada fila representa una observación (el cliente) y cada columna representa una característica.

Ingeniería de Características: El proceso de unión no es solo un merge; es una transformación de series de tiempo/transaccional (múltiples filas de historial) en características de resumen (una sola fila con promedios, sumas y conteos). Por ejemplo, el número promedio de días de mora o la suma total de deuda externa son métricas clave de riesgo que solo se obtienen al resumir las tablas bureau y previous_application.

El One-Hot Encoding (OHE) se aplicó principalmente a las variables categóricas de las tablas secundarias (como STATUS en bureau_balance o NAME_CONTRACT_STATUS en previous_application).

Conservación de la Información Categórica: El OHE convierte una variable categórica (ej., Crédito Activo) en varias columnas binarias (0 o 1) , lo que permite que el modelo las use.

En el Contexto de Agregación (Clave para el Negocio): Cuando agregamos a nivel de cliente, el OHE nos permite transformar una columna categórica en conteo de ocurrencias a nivel de cliente.

En resumen, la unión de tablas y el OHE fueron pasos técnicos obligatorios para traducir la información de riesgo dispersa y cualitativa en métricas numéricas y estructuradas que el modelo puede aprender y utilizar para predecir el score.

# Modelado y evaluacion
En el modelado empezaremos por PCA para simplificar el dataset de 913 columnas y reducir el ruido antes de aplicar las otras técnicas. Tanto K-Means como Isolation Forest dependen de la distancia y el escalado de las variables. Aplicar K-Means o Isolation Forest en un espacio de 913 dimensiones puede generar resultados sesgados, inestables o demasiado lentos.

# PCA
Al aplicar PCA primero, se obtendran nuevas variables no correlacionadas (Componentes Principales) que capturan la mayor variación, lo que hace que los clusters de K-Means y los outliers de Isolation Forest sean más significativos y eficientes de calcular.
Se necesitaron 192 componentes de las 475 iniciales para explicar el 90% de la varianza total de los datos.
El DataFrame reducido (df_pca_final) tiene: 192 columnas con las que se trabajara el k-means.

# K-means
Luego con K-means se determinara el número óptimo de clusters (K). Se usara el Método del Codo (Elbow Method) para encontrar este valor. Para calcular la Inercia para el Método del Codo se uso un rango de entre 2 a 11 para el K.
El K optimo estimado fue de 5 (donde se observa la curva). Al calcular la tasa de incumplimiento de llego a los siguientes numeros:
  KMEANS_CLUSTER |   Total Clientes |   Tasa de Incumplimiento Promedio |
|-----------------:|-----------------:|----------------------------------:|
|                0 |           112998 |                         0.0764792 |
|                1 |            13866 |                         0.0512765 |
|                2 |            25701 |                         0.0937318 |
|                3 |            23940 |                         0.0773601 |
|                4 |           131006 |                         0.0855762 |

--- Ingreso Promedio por Cluster ---
|    |   KMEANS_CLUSTER |   Ingreso Promedio |
|---:|-----------------:|-------------------:|
|  0 |                0 |             170206 |
|  1 |                1 |             210804 |
|  2 |                2 |             187049 |
|  3 |                3 |             196478 |
|  4 |                4 |             154498 |

Donde se muestra que la tasa de incumplimiento global es de un 8.07%. 
Ll grupo 4 es el que tiene mayores numeros de clientes con una tasa moderada de incumplimiento.
El grupo 2 es el de mayor riesgo con un 9,37% de incumplimiento, un 1,6% mas que el global, a pesar de tener mayores ingresos que los grupos 0 y 4, esto sugiere que el riesgo no es por pobreza, si no posiblemente por altas deudas externas.
Mientras que la mas baja es el grupo 1 con un 5,13%, siendo tambien el grupo con menos clientes y el ingreso mas alto, el grupo ideal para trabajar con modelos supervisados mas flexibles.
El grupo que mas se deberia tener en cuenta es el 4, pues es el que mayor cantidad de clientes tiene.
Podemos usar el metodo de la silueta para validar la eleccion del K, sin embargo se hara solo con el 10% de las muestras, ya que, este metodo toma muchos recursos y tiempo. Lo que se muestra con este metodo es que el K optimo es 2, debido a que con el 10% de las muestras solo se logran distinguir los grupos mas densos de informacion, esto es porque los datos estan muy dispersos aun aplicando el PCA, pues una muestra pequeña en un espacio de 150 dimenciones es menos probable que represente la verdadera varianza del dataset completo que una muestra del 90%.
Tambien esta el metodo del dendrograma que, de manera visual, se puede identificar el K optimo. Se opto por usar 1000 muestras agrupadas en los 10 clusters principales, se nos muestra que segun la distancia Ward el numero optimo seria entre el 150 y 200, lo que daria un K optimo = 3 o 5.
El problema de este metodo es que no es posible usar los datos completos por sobrecarga de los datos, lo que puede ser poco representativo.

# Isolation forest
Finalmente con Isolation Forest se detectaran los outliers, donde detectaremos la tasa de incumplimiento entre las observaciones normales y las anomalas.
En la tabla se muestran la cantidad de clientes con registros anomalos (-1) y los normales(1) 
--- Análisis de Riesgo: Normal vs. Anomalía ---
| ISOLATION_OUTLIER   |   Total_Clientes | Tasa_Incumplimiento_Promedio   |
|:--------------------|-----------------:|:-------------------------------|
| Anomalía            |             3076 | 9.79%                          |
| Normal              |           304435 | 8.06%                          |

Se muestra que los registros de clientes anomalos superan a los normales con un 9,79% respecto a un 8.06%, lo que sugiere este comportamiento inusual es una correlaciona con un riesgo extremo. Esto indicaría que el modelo puede estar detectando clientes de alto riesgo no capturados fácilmente por los features promedio.

# Evaluacion final
En esta ultima seccion se describiran las tecnicas utilizadas y porque se eligieron.
Este proyecto complementa el modelo de scoring crediticio supervisado mediante la aplicación de un pipeline secuencial de tres técnicas no supervisadas sobre el conjunto de entrenamiento, con el objetivo de mejorar la estabilidad, eficiencia, y la comprensión de los segmentos de riesgo.

Técnica Principal Implementada: Análisis de Clusters (K-Means).
Técnicas Complementarias: Reducción de Dimensionalidad (PCA) y Detección de Anomalías (Isolation Forest).

Justificación de la Elección del Pipeline
La estrategia elegida fue aplicar las técnicas en el siguiente orden para optimizar el análisis:

1°-PCA (Análisis de Componentes Principales): Se aplicó primero para reducir el ruido y la alta correlación entre las más de 900 variables del dataset fusionado. Esto crea un espacio de características más estable para K-Means y acelera el entrenamiento posterior. El dataset transformado por PCA (df_pca_final) debe ser utilizado como base para entrenar el modelo supervisado principal. Esto reduce el tiempo de entrenamiento y mitiga el riesgo de multicolinealidad, aumentando la estabilidad del modelo.

K-Means: Se aplicó sobre el dataset reducido por PCA para identificar segmentos homogéneos de clientes. Esto permite analizar si el riesgo de incumplimiento se distribuye de manera desigual entre grupos específicos, identificando potenciales sesgos o subpoblaciones. La columna categórica KMEANS_CLUSTER debe incorporarse al dataset de entrenamiento del modelo supervisado. Al darle al modelo información explícita sobre el segmento al que pertenece un cliente (ej., "pertenece al Clúster 2 de Alto Riesgo Oculto"), se espera que mejore la capacidad predictiva del modelo, especialmente en las fronteras de decisión.

Isolation Forest: Se utilizó para detectar observaciones atípicas (outliers). El propósito es evaluar si la exclusión o el tratamiento diferenciado de estos casos podría mejorar la estabilidad del modelo final. Los clientes marcados como Anomalía (-1) deben ser señalados.