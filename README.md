# Modelos de clasificación

En los archivos explico la mayoría de las líneas de código, sin embargo aquí te explicaré de una manera más detallada.

Primero, cuando estés analizando el dataset te darás cuenta que algunas **variables son categóricas**, como el sexo de un paciente, y tú deseas cambiarla a una **variable numérica** de tipo Int, así que puedes elegir varios caminos para esto, hay algunos más eficientes y apropiados, esto también va de la mano del tipo de modelo y datos que estás implementando.

Una función que puedes utilizar es **LabelEncoder()** que se encuentra en el módulo `from sklearn import preprocessing`, en conjunto se usa con la librería pandas para codificar las variables, usemos de ejemplo una variable llamada sex que se encuentra en un dataset llamado train_df.

```sh
label_encoder = preprocessing.LabelEncoder()
encoder_sex = label_encoder.fit_transform(train_df['Sex'])
```

Si estás empezando, puedes hacerlo de una menera un poco más primitiva, aunque podrás tener inconvenientes si tiene muchas categorías

```sh
test_df['Sex'].replace('male',1,inplace=True)
test_df['Sex'].replace('female',0,inplace=True)
```

Otra manera de hacerlo es con **get_dummie** que a mi parecer es de las mejores. Es una función de Pandas que se utiliza para crear variables ficticias (también llamadas "variables dummy") a partir de variables categóricas.

`variable = pd.get_dummies(features)`

get_dummies utiliza **One-Hot Encoding**, es una técnica de transformación de variables categóricas en variables numéricas que consiste en crear una nueva columna para cada categoría de la variable original y asignar un valor binario (1 o 0) a cada fila de la nueva columna en función de si la fila pertenece o no a la categoría correspondiente, si pertenece le dará un valor de 1, sino le dará un 0.

LabelEncoder() al contrario de get_dummies puede ser un poco confuso para el modelo porque digamos que tenemos 3 categorías y a cada una le asigna un valor de 1, 2, 3. Esto le puede sonar al modelo que las variables tienen importancia en función del tamaño u orden de sus números.

Si tu target o variable de salida es una variable categórica y necesitas que sea una variable binaria, puedes hacerlo mediante un mapeado con el método **.map()**.

Hagamos de cuenta que nuestra variable de salida se llama `Enfermedad` y está en nuestro dataset `datos`:

datos['Enfermedad'] = datos['Enfermedad'].map({'SI': 1, 'NO': 0})

Así nuestra variable quedaría numérica y binaria para que nuestro modelo pueda trabajar más facilmente ya que muchos de estos algoritmos están optimizados para trabajar con variables de este tipo.

Ahora bien, resulta que hay variables con valores nulos, esto podemos averiguarlo con un simple código `train_df.isnull().any()`, es importante saber que para implementar modelos de Machine learning las variables estén completas y no tengan valores faltantes por las siguientes razones.

1. La precisión del modelo se verá afectada: si hay muchos valores nulos en los datos, el modelo puede tener dificultades para encontrar patrones y relaciones entre las variables. Esto puede afectar negativamente la precisión del modelo y hacer que produzca predicciones inexactas.
2. La calidad de los datos se verá comprometida: los valores nulos pueden ser un indicador de que los datos tienen problemas de calidad o que no se han recopilado de manera adecuada
Entonces para dar solución a este problema puedes utilizar alternativas simples como utilizar el método **.fillna**: 

```sh
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
```

Se pueden llenar los datos con la media o la mediana, pero ten en cuenta que pueden haber outliers o valores atípicos, entonces la media podría verse afectada por estos y no representar adecuadamente la tendencia central de los datos, y si los valores núlos se completan de manera incorrecta estos pueden comprometer aún más la calidad de los datos y llevar a conclusiones erróneas.

***Árbol de Decisión gráfico***

Otra duelo para entender es la imágen de un Árbol de decisión, ¿Gini, samples, value, class?. No soy un experto en el tema pero espero mi explicación al menos despeje tus dudas o sea una buena partida para entenderlo.

Empecemos por hablar de cada parte de un árbol.

* Nodo Raiz: Es el primer rectángulo que aparece, no tiene un nodo padre, y de ahí se inicia a desprender nodos hijos.

* Hojas: Son los últimos nodos del árbol, cada hoja representa una clasificación o resultado final, es decir una decisión tomada por el árbol.

* Gini: el índice Gini es una medida de la impureza de un nodo, la impureza se refiere a cuán mezcladas están las clases en un conjunto de datos. Si todos los ejemplos de una hoja pertenecen a la misma clase, entonces se dice que la hoja es "pura". Por otro lado, si hay una mezcla de clases en una hoja, entonces se dice que la hoja es "impura". Cuanto más cercano a cero sea el índice Gini, más pura será la hoja y mejor será la separación de clases, Si el índice Gini fuera de 0.5 en un nodo, esto indicaría que hay una distribución uniforme de los ejemplos entre las clases posibles.

* Samples: Ejemplos que cumplen la condición inicial.

* Value: el parámetro "value" en un nodo de un árbol de decisión representa la distribución de los valores de la variable objetivo (la variable que se quiere predecir) en el conjunto de ejemplos que están asignados a ese nodo.

Supongamos que tenemos un nodo del dataset Titanic con los siguientes valores:

```sh
Pclass <= 0.229
gini = 0.383
samples = 314
Value = [81,233]
```

Pclass podría tener los posibles valores [0.82, -1.15, -0.33]

Entonces el nodo se pregunta ¿Qué pasajeros tienen una Pclass menor o igual a 0.229?, bueno si miramos los posibles valores hay dos clases que entrarían.
el gini nos dice que es de 0.383, la impureza no es baja pero es considerada es decir que hay más de una clase, en efecto ya lo habíamos mirado.
me dice que hay 314 pasajeros que cumplen la condición Pclass, y de esos 314 hay 81 que no sobrevivieron mientras que 233 sí lo hicieron, esta información nos lo proporciona el value = [81,233].

Bien, cuando se evaluó la condición de Pclass dio verdadera para 314 pasajeros, así que estos se irán por la rama izquiera a evaluar una nueva condición y siguir dividiéndose mediante los nodos hijos, mientras para los que la condición fue falsa seguirán la rama derecha.

Espero haber sido de ayuda, lo intenté explicar de una manera no tan técnica, si quieres retroalimentar mi explicación no dudes en escribirme, estaré atento.
