#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sched, time

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import StreamingKMeans

if __name__ == '__main__':
    # Cria os contextos do Spark e Spark Streaming com batch interval de 10s
    sc = SparkContext(appName='StreamingKMeansClustering')
    ssc = StreamingContext(sc, 10)

    # Cria um checkpoint e salva os arquivos no diretório informado
    ssc.checkpoint('file:///tmp/spark')

    # Extrai a latitude e longitude dos dados de treinamento e 
    # transforma em um vetor
    def parse_training_data(line):
        cells = line.split(',')
        vec = Vectors.dense([float(cells[0]), float(cells[1])])
        return vec

    # Procura por arquivos de texto na pasta informada.
    # Por se tratar de Stream, o Spark irá monitorar constantemente 
    # esse diretório e qualquer arquivo adicionado ao diretório,
    # será incluído nos dados de treinamento. Cada arquivo adicionado,
    # é transformado em um novo batch da stream.
    # A função map, irá fazer executar a função parseTrainingData para
    # realizar o parse dos dados do arquivo para a stream de treinamento.
    training_stream = ssc.textFileStream(
        'file:///home/andrey/Documentos/Projects/streaming-k-means/training/')\
        .map(parse_training_data)            

    # Inicializa o algoritmo k-means com streaming para rodar sobre os dados
    # adicionados ao diretório de streaming.
    # k=2: Número de clusters em que o dataset será dividido
    # decayFactor=1.0: Todos os dados, desde o início, são relevantes.
    #             0.0: Utilização somente dos dados mais recentes.
    # O k-means requer o centro dos clusters randômicos para iniciar o
    # processo:
    # 2: Quantidade de centros a serem setados
    # 1.0 e 0: weight e seed
    model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(2, 1.0, 0)

    # Imprime os centros.
    print('Initial centers: ' + str(model.latestModel().centers))

    # Treinamento do modelo
    model.trainOn(training_stream)

    # Inicia a stream
    ssc.start()

    # Agenda a impressão dos valores do centros em tempos periódicos
    s = sched.scheduler(time.time, time.sleep)

    # Função que imprime os centros recursivamente, a cada 10s.
    def print_cluster_centers(sc, model):
        print('Cluster centers: ' + str(str(model.latestModel().centers)))
        s.enter(10, 1, print_cluster_centers, (sc, model))

    # A função para imprimir os clusters (print_cluster_centers) será 
    # executada a cada 10s com prioridade 1. Essa função aceita dois 
    # argumentos, o schedule s e o modelo representado pela variável
    # model
    s.enter(10, 1, print_cluster_centers, (s, model))
    s.run()

    # Executa a aplicação até que um comando de término seja informado
    ssc.awaitTermination()