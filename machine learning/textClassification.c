#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAX_LENGTH 50
#define MAX_WORDS 1500
#define NOKTALAMA " .,;:!?()\n"
#define LEARNING_RATE 0.03
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

int uniqueWord(char *, char dictionary[][MAX_LENGTH], int, int *);
void removePunctation(char*);
void toLowerLetter(char*);
void splitTextToWord(char *, char dictionary[][MAX_LENGTH], int *, int *, int, int*);
void addSpace(char*, int);
void randomShuffle(int*, int*, int);

int main() {
    srand(time(NULL));

    char *text1;
    char *text2;
    char dictionary[MAX_WORDS * 2][MAX_LENGTH];
    int count = 0;
    int i, j, k, epoch = 1000, num;
    int *vectors; // tum metinlerin hot vectorleri
    int *y; // etiketler

    text1 = (char*)malloc(MAX_WORDS * MAX_LENGTH * sizeof(char));
    text2 = (char*)malloc(MAX_WORDS * MAX_LENGTH * sizeof(char));
    y = (int*)malloc(MAX_WORDS * 2 * sizeof(int));

    FILE *file;
    file = fopen("biyoloji.txt", "r");
    if (file == NULL) {
        printf("\nFile opening error!");
        return 1;
    }
    fgets(text1, MAX_WORDS * MAX_LENGTH, file);
    int uzunluk = strlen(text1);
    addSpace(text1, uzunluk);
    
    file = fopen("fizik.txt", "r");
    if (file == NULL) {
        printf("\nFile opening error!");
        return 1;
    }
    fgets(text2, MAX_WORDS * MAX_LENGTH, file);
    uzunluk = strlen(text2);
    addSpace(text2, uzunluk);
    fclose(file);

    vectors = (int*)calloc(MAX_WORDS * 2 * MAX_WORDS * 2, sizeof(int));
    splitTextToWord(text1, dictionary, &count, vectors, 0, y);
    splitTextToWord(text2, dictionary, &count, vectors, 1, y);
    free(text1);
    free(text2);

    FILE *fp = fopen("loss_data.csv", "w");
    if (fp == NULL) {
        printf("\nFile opening error!");
        return 1;
    }

    fprintf(fp, "epoch,loss,time,method,w_value\n");

    FILE *wGD_file = fopen("wGD_values.csv", "w");
    if (wGD_file == NULL) {
        printf("\nFile opening error!");
        return 1;
    }

    fprintf(wGD_file, "epoch");
    for (j = 0; j < count; j++) {
        fprintf(wGD_file, ",wGD%d", j + 1);
    }
    fprintf(wGD_file, "\n");

    FILE *wSGD_file = fopen("wSGD_values.csv", "w");
    if (wSGD_file == NULL) {
        printf("\nFile opening error!");
        return 1;
    }

    fprintf(wSGD_file, "epoch");
    for (j = 0; j < count; j++) {
        fprintf(wSGD_file, ",wSGD%d", j + 1);
    }
    fprintf(wSGD_file, "\n");

    FILE *wADAM_file = fopen("wADAM_values.csv", "w");
    if (wADAM_file == NULL) {
        printf("\nFile opening error!");
        return 1;
    }

    fprintf(wADAM_file, "epoch");
    for (j = 0; j < count; j++) {
        fprintf(wADAM_file, ",wADAM%d", j + 1);
    }
    fprintf(wADAM_file, "\n");

    for (num = 0; num < 5; num++) {
        printf("\n%d.w degerleri icin:", num + 1);
        double *w;
        w = (double*)malloc(count * sizeof(double));
        for (i = 0; i < count; i++) {
            w[i] = (((double)rand() / (double)RAND_MAX) * 2.0) - 1.0; // -1 ile 1 arasinda rastgele float sayilar atanir
        }
        double *wGD = (double*)malloc(count * sizeof(double));
        double *wSGD = (double*)malloc(count * sizeof(double));
        double *wADAM = (double*)malloc(count * sizeof(double));
        for (i = 0; i < count; i++) {
            wGD[i] = w[i];
            wSGD[i] = w[i];
            wADAM[i] = w[i];
        }

        int train_num = count * 0.8;
        int test_num = count * 0.2;
        int choice;
        randomShuffle(vectors, y, count);

        // Gradient Descent
        for (k = 0; k < epoch; k++) {
            double gradients[count];
            for (j = 0; j < count; j++) {
                gradients[j] = 0.0;
            }
            double start_time = (double)clock() / CLOCKS_PER_SEC;
            double loss = 0.0;
            for (i = 0; i < train_num; i++) {
                double y_pred = 0.0;
                for (j = 0; j < count; j++) {
                    y_pred += wGD[j] * vectors[i * count + j];
                }
                y_pred = tanh(y_pred);
                double error = y[i] - y_pred;
                loss += error * error;
                for (j = 0; j < count; j++) {
                    gradients[j] += error * vectors[i * count + j];
                }
            }
            loss /= train_num;
            double end_time = (double)clock() / CLOCKS_PER_SEC;
            double epoch_time = end_time - start_time;
            fprintf(fp, "%d,%f,%f,GD,%d\n", k + 1, loss, epoch_time, num + 1);
            fprintf(wGD_file, "%d", k + 1);
            for (j = 0; j < count; j++) {
                wGD[j] += LEARNING_RATE * gradients[j] / train_num;
                fprintf(wGD_file, ",%f", wGD[j]);
            }
            fprintf(wGD_file, "\n");
        }
        double accuracy = 0.0;
        for (i = train_num; i < count; i++) {
            double y_pred = 0.0;
            for (j = 0; j < count; j++) {
                y_pred += wGD[j] * vectors[i * count + j];
            }
            y_pred = tanh(y_pred);
            if ((y_pred >= 0.0 && y[i] == 1) || (y_pred < 0.0 && y[i] == -1)) {
                accuracy += 1.0;
            }
        }
        accuracy = (accuracy / test_num) * 100.0;
        printf("\nGD Accuracy: %.2f%%\n", accuracy);  
		
        // Stochastic Gradient Descent
        for (k = 0; k < epoch; k++) {
            double start_time = (double)clock() / CLOCKS_PER_SEC;
            double loss = 0.0;
            for (i = 0; i < train_num; i++) {
                double y_pred = 0.0;
                for (j = 0; j < count; j++) {
                    y_pred += wSGD[j] * vectors[i * count + j];
                }
                y_pred = tanh(y_pred);
                double error = y[i] - y_pred;
                loss += error * error;
                for (j = 0; j < count; j++) {
                    wSGD[j] += LEARNING_RATE * error * vectors[i * count + j];
                }
            }
            loss /= train_num;
            double end_time = (double)clock() / CLOCKS_PER_SEC;
            double epoch_time = end_time - start_time;
            fprintf(fp, "%d,%f,%f,SGD,%d\n", k + 1, loss, epoch_time, num + 1);
            fprintf(wSGD_file, "%d", k + 1);
            for (j = 0; j < count; j++) {
                fprintf(wSGD_file, ",%f", wSGD[j]);
            }
            fprintf(wSGD_file, "\n");
        }
        accuracy = 0.0;
        for (i = train_num; i < count; i++) {
            double y_pred = 0.0;
            for (j = 0; j < count; j++) {
                y_pred += wSGD[j] * vectors[i * count + j];
            }
            y_pred = tanh(y_pred);
            if ((y_pred >= 0.0 && y[i] == 1) || (y_pred < 0.0 && y[i] == -1)) {
                accuracy += 1.0;
            }
        }
        accuracy = (accuracy / test_num) * 100.0;
        printf("\nSGD Accuracy: %.2f%%\n", accuracy);

        // Adam Optimization
        double m[count], v[count];
        for (j = 0; j < count; j++) {
            m[j] = 0.0;
            v[j] = 0.0;
        }
        for (k = 0; k < epoch; k++) {
            double gradients[count];
            for (j = 0; j < count; j++) {
                gradients[j] = 0.0;
            }
            double start_time = (double)clock() / CLOCKS_PER_SEC;
            double loss = 0.0;
            for (i = 0; i < train_num; i++) {
                double y_pred = 0.0;
                for (j = 0; j < count; j++) {
                    y_pred += wADAM[j] * vectors[i * count + j];
                }
                y_pred = tanh(y_pred);
                double error = y[i] - y_pred;
                loss += error * error;
                for (j = 0; j < count; j++) {
                    gradients[j] += error * vectors[i * count + j];
                }
            }
            loss /= train_num;
            double end_time = (double)clock() / CLOCKS_PER_SEC;
            double epoch_time = end_time - start_time;
            fprintf(fp, "%d,%f,%f,ADAM,%d\n", k + 1, loss, epoch_time, num + 1);
            fprintf(wADAM_file, "%d", k + 1);
            for (j = 0; j < count; j++) {
                m[j] = BETA1 * m[j] + (1 - BETA1) * gradients[j];
                v[j] = BETA2 * v[j] + (1 - BETA2) * gradients[j] * gradients[j];
                double m_hat = m[j] / (1 - pow(BETA1, k + 1));
                double v_hat = v[j] / (1 - pow(BETA2, k + 1));
                wADAM[j] += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
                fprintf(wADAM_file, ",%f", wADAM[j]);
            }
            fprintf(wADAM_file, "\n");
        }
        accuracy = 0.0;
        for (i = train_num; i < count; i++) {
            double y_pred = 0.0;
            for (j = 0; j < count; j++) {
                y_pred += wADAM[j] * vectors[i * count + j];
            }
            y_pred = tanh(y_pred);
            if ((y_pred >= 0.0 && y[i] == 1) || (y_pred < 0.0 && y[i] == -1)) {
                accuracy += 1.0;
            }
        }
        accuracy = (accuracy / test_num) * 100.0;
        printf("\nADAM Accuracy: %.2f%%\n", accuracy);

        free(w);
        free(wGD);
        free(wSGD);
        free(wADAM);
    }

    fclose(fp);
    fclose(wGD_file);
    fclose(wSGD_file);
    fclose(wADAM_file);
    free(vectors);
    free(y);
    return 0;
}

int uniqueWord(char *word, char dictionary[][MAX_LENGTH], int count, int *index){
    int i;
    for(i=0; i<count; i++){
        if (strcmp(dictionary[i], word) == 0) {
            *index=i;
            return 1; // Kelime bulundu
        }
    }    
    return 0;
}

void removePunctation(char* str) {
    int len = strlen(str);
    int i,j;
    for (i = 0; i < len; i++) {
        if (ispunct(str[i])) {
            for (j = i; j < len - 1; j++) {
                str[j] = str[j + 1];
            }
            str[len - 1] = '\0';
            len--;
            i--;
        }
    }
}

void toLowerLetter(char* str) {
    int i;
    for (i = 0; str[i] != '\0'; i++) {
        str[i] = tolower(str[i]);
    }
}

void splitTextToWord(char *metin, char dictionary[][MAX_LENGTH], int *count, int *vectors, int hangisi, int *y) {
    char kopyaMat[MAX_WORDS * MAX_LENGTH];
    strcpy(kopyaMat, metin);
    int index;
    char *token = strtok(kopyaMat, NOKTALAMA);
    while (token != NULL && (*count) < MAX_WORDS * 2) {
        toLowerLetter(token);   
        if (!uniqueWord(token, dictionary, *count, &index)) {
            strcpy(dictionary[*count], token);
            index = *count;
            (*count)++;
        }
        vectors[hangisi * MAX_WORDS + index] = 1;
        y[hangisi * MAX_WORDS + index] = hangisi == 0 ? 1 : -1;
        token = strtok(NULL, NOKTALAMA);   
    }
}

void addSpace(char*metin, int uzunluk){// Oldugu metinde son kelimeyi alabilsin diye bosluk ekledim
    if (uzunluk > 0 && metin[uzunluk - 1] == '\n') {
        metin[uzunluk-1] = ' ';
        metin[uzunluk] = '\0';
    }
}

void randomShuffle(int *vectors, int *y, int count) {
    int i, j,k, temp;
    for (i = 0; i < count; i++) {
        j = rand() % count;
        for (k = 0; k < MAX_WORDS * 2; k++) {
            temp = vectors[i * MAX_WORDS * 2 + k];
            vectors[i * MAX_WORDS * 2 + k] = vectors[j * MAX_WORDS * 2 + k];
            vectors[j * MAX_WORDS * 2 + k] = temp;
        }
        temp = y[i];
        y[i] = y[j];
        y[j] = temp;
    }
}

