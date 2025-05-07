# Makefile para flujo de trabajo ML local

# Limpieza de datos
clean:
	python clean_data.py

# Exploración de datos (genera gráficas en outputs/)
explore:
	python explore_data.py

# Entrenamiento de modelos (guarda mejor modelo en models/)
train:
	python train.py

# Validación final sobre el conjunto de test
validate:
	python validate.py

# Ejecutar todo en orden
full:
	make clean
	make explore
	make train
	make validate
