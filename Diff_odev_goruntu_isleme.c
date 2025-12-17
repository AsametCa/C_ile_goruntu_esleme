#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>
#include <conio.h>

void saveResultsToCSV(const char *filename, int method, double startVal, 
                       double *losses, double *trainAcc, double *testAcc, double *times, int count, int j, double **wVal);
void normalizeAndSave(double **matrix, int index, unsigned char *img, int width, int height, int channels);
double makePredict(double *w, double * x, int N );
double  gradient_descent(double ** data ,double *w , int size , int N , double stepS , double*  trainA);
double Stochastic_Gradient(double **data , double *w,int size , int N,double stepS, double * trainA);
double adam(double **data,double * w,double *m, double *v , double *mVar, double *vVar, int size , int N, double stepS , int count , double* trainA);
double testFonc(double ** test, double *w, int N , int size , int index );


double makePredict(double *w, double * x, int N ){
    double predict =0.0;
    int i =0;
    
   
    for( i = 0; i< N*N+1; i++){
        predict += w[i]*x[i];

    }
    return tanh(predict);
}
 double  gradient_descent(double ** data ,double *w , int size , int N , double stepS , double*  trainA){
  
    double error = 0.0,  loss=0.0 , prediction , train = 0; ;
    int label = 1;
    int j;
    double * gradian = (double*)malloc((N*N+1) *sizeof(double));
   
    for(int j = 0; j<N*N+1;j++){
        gradian[j] = 0.0;
    }
    
    for(int i=0; i< size-1 ; i++)
    {   
        
        if(i<=size/2){
            label = -1;
        }
        else {
            label = 1;
        }
        prediction = makePredict(w, data[i], N);
       
        train += prediction/label;
        
        error = label - prediction;
        
        loss += error * error;
       
        for(j=0 ; j < N*N +1 ; j++){
            gradian[j] += (-1* 2* error *(1- prediction*prediction)*data[i][j]);
        }
       
    }
    
    for(int i = 0; i< N*N+1; i++){
        
         w[i] -= gradian[i]*stepS/size;
        
    }
    *trainA = train/size;
    loss /=size;
    free(gradian);
    return loss;
 } 
 double Stochastic_Gradient(double **data , double *w,int size , int N,double stepS, double * trainA){
    double loss = 0.0, error , prediction , train = 0.0;
    int label ;
    int j;
    double * gradian = (double*)malloc((N*N+1) *sizeof(double));
    
   
    for(int j = 0; j<N*N+1;j++){
        gradian[j] = 0.0;
    }
   
    int i;
    for(int k =0 ; k<300;k++){
    i=rand()%(size-1);
        if(i<=size/2){
            label = -1;
        }
        else {
            label = 1;
        }
        prediction = makePredict(w, data[i], N); 
        train += prediction /label ;     
        error = label - prediction;
        loss += error * error;
        
       
        for(j=0 ; j < N*N +1 ; j++){
            gradian[j] = (-1* 2* error *(1- prediction*prediction)*data[i][j]);
            w[j] -= gradian[j]*stepS;
        }
    }
    *trainA = train/300;
     
    
    free(gradian);
    return loss/300;

 }

 double adam(double **data,double * w,double *m, double *v , double *mVar, double *vVar, int size , int N, double stepS , int count , double* trainA){
    double firstM = 0.9,secondM = 0.999 , control = 0.00000001 , predict,loss =0.0,error , train =0 ;
    double *gradian ;
    gradian = (double*)malloc((N*N+1)*sizeof(double));
    

    for(int j =0 ; j< N*N+1; j++){
        m[j] = 0.0;
        v[j] =0.0;
        gradian[j] =0.0;
    }
    int label = 1;
    count = 1;
    
    for(int k=0; k<200;k++){
    int i =rand()%(size-1);
    
        if(i<size/2){
            label = -1;
        }
        else{
            label= 1;
        }
        predict=makePredict(w,data[i],N);
        train += predict/label;
        error = label -predict;
        loss += error*error;
        for (int j = 0; j < N*N+1; j++)
        {
            gradian[j] = (-2* error *(1- predict*predict)*data[i][j]);
            m[j] = m[j]*firstM+(1-firstM)*gradian[j];
            v[j] = v[j]*secondM+(1-secondM)*gradian[j]*gradian[j];
            mVar[j] = m[j]/(1-pow(firstM,count));
            vVar[j] = v[j]/(1-pow(secondM,count));
             w[j] -= (mVar[j]*stepS/(sqrt(vVar[j])+control));
        }
        count++;    
    }    
    *trainA = train/200;
    free(gradian);
    return loss/200;
    
 }
 double testFonc(double ** test, double *w, int N , int size , int index ){
    double total =0.0 , accurate =  0.0;
      int label = 1;
    if(index<size/2){
        label =1;
    }
    else{
        label=-1;
    }
    for(int i = 0; i<N*N+1; i++){
        total += w[i]*test[index][i];
    }
    
    accurate = tanh(total)/label;
    return accurate;
}

void saveResultsToCSV(const char *filename, int method, double startVal,
                         double *losses, double *trainAcc, double *testAcc, double *times, int count , int j,double **wVal) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }
    FILE *file2 = fopen("wVal.txt","w+");
    int val = 0;
   
    
    for (int i = 0; i < count-1; i++) {
        fprintf(file, "%d,",i );
    }
    fprintf(file, "%d",count-1 );
    fprintf(file, "\n");   
    for (int i = 0; i < count-1; i++) {
        fprintf(file, "%lf,", losses[i]);
    }
    fprintf(file, "%lf", losses[count-1]);
    fprintf(file, "\n");
    for (int i = 0; i < count-1; i++) {
        fprintf(file,"%lf,", trainAcc[i]);
    }
    fprintf(file,"%lf", trainAcc[count]);
    fprintf(file, "\n");

    
    for (int i = 0; i < count-1; i++) {
        fprintf(file, "%lf,", testAcc[i]);
    }
    fprintf(file, "%lf", testAcc[count-1]);
    fprintf(file, "\n");

   
    for (int i = 0; i < count-1; i++) {
        fprintf(file, "%.4lf,", times[i]);
    }
    fprintf(file, "%.4lf", times[count-1]);
    fprintf(file, "\n\n");

    for (int i = 0; i < count-1; i++) {
       for (int k = 0; k < 28*28+1; k++)
       {
              fprintf(file2, "%lf,", wVal[i][k]);
       }
       
            fprintf(file2, "init%d",j);
         fprintf(file2, "\n");
       
    }
    fprintf(file, "%lf", wVal[count-1]);

    fclose(file);
}
void normalizeAndSave(double **matrix, int index, unsigned char *img, int width, int height, int channels) {
    int flat_index = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char pixel = img[y * width * channels + x * channels];
            double normalized = pixel / 255.0; // Normalizasyon
            matrix[index][flat_index] = normalized; // Matrise yaz
            flat_index++;
        }
    }

    matrix[index][flat_index] = 1.0; // Bias
}


int main() {
    const char *input_folder = "C:/Users/SAMET/Odev/Image2";
    const char *txt_fileName = "data";
   
   srand(time(0));
    // Yeni boyutlar
    int new_width = 28;
    int new_height = 28;
    double stepS = 0.0001;
    int initial_size = 800;
         double **data = (double **)malloc(initial_size * sizeof(double *));
    double **test = (double **)malloc(initial_size * sizeof(double *));
    for (int i = 0; i < initial_size; i++) {
        data[i] = (double *)malloc((new_width * new_height + 1) * sizeof(double));
        test[i] = (double *)malloc((new_width * new_height + 1) * sizeof(double));
    }
    double *w =(double*)malloc((new_width * new_height + 1) * sizeof(double));
    for(int i =0;i<new_height*new_width;i++){
        w[i]=0.01;
    }
    w[new_height*new_width] = 1;

    // Resimleri dosyadan al ve kümelere yaz
    DIR *dir;
    struct dirent *ent;
    int data_index = 0, test_index = 0;
    int counter = 0;
    

    if ((dir = opendir(input_folder)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                char input_file[256];
                sprintf(input_file, "%s/%s", input_folder, ent->d_name);

                // Resmi yükle
                int width, height, channels;
                unsigned char *img = stbi_load(input_file, &width, &height, &channels, 0);
                if (img == NULL) {
                    printf("Error loading image: %s\n", input_file);
                    continue;
                }

                // Normalize et ve uygun kümeye yaz
                if (counter <3200) { // %80: Eğitim
                    if (data_index >= initial_size) {
                        initial_size *= 2;
                        data = (double **)realloc(data, initial_size * sizeof(double *));
                        for (int i = data_index; i < initial_size; i++) {
                            data[i] = (double *)malloc((new_width * new_height + 1) * sizeof(double));
                        }
                    }
                    normalizeAndSave(data, data_index++, img, new_width, new_height, channels);
                } else { // %20: Test
                    if (test_index >= initial_size) {
                        initial_size *= 2;
                        test = (double **)realloc(test, initial_size * sizeof(double *));
                        for (int i = test_index; i < initial_size; i++) {
                            test[i] = (double *)malloc((new_width * new_height + 1) * sizeof(double));
                        }
                    }
                    normalizeAndSave(test, test_index++, img, new_width, new_height, channels);
                }

                // Resim verisini serbest bırak
                stbi_image_free(img);
                counter++;
            }
        }
        closedir(dir);
    } else {
        printf("Could not open directory %s\n", input_folder);
        return EXIT_FAILURE;
    }
    int max_iterNum = 200;
  
    printf("Total Images Processed: %d\n", counter);
    printf("Training Set Size: %d\n", data_index);
    printf("Test Set Size:  %d\n", test_index);
    clock_t start_time, end_time;
    double time;
    int epoch =0;
    double loss = 10.0;
    double stopValue = 100;
    double trainA = 2;
    stepS = 0.1;
    double startVal = 0.001;
    double  *lossptr, *trainptr , *testptr,*timeptr, **wVal;

    wVal = (double**)malloc(max_iterNum*sizeof(double*));
    for(int i = 0; i<max_iterNum; i++){
        wVal[i] = (double*)malloc((new_height*new_width+1)*sizeof(double));
    }
    if(wVal == NULL){
        printf("Memory Allocation Fault");
        exit(EXIT_FAILURE);
    }
    lossptr =(double*)malloc(max_iterNum*sizeof(double));
    trainptr =(double*)malloc(max_iterNum*sizeof(double));
    testptr=(double*)malloc(max_iterNum*sizeof(double));
    timeptr =(double*)malloc(max_iterNum*sizeof(double));

    if(lossptr == NULL ||trainptr == NULL||testptr == NULL||timeptr ==NULL){
        printf("Memory Allocation Fault");
        exit(EXIT_FAILURE);
    }
    char output_file[256];
    printf("merhaba\n");
    double totalAccurate = 1;
   for(int j = 0 ; j<5; j++){
        
        loss =100;
        stepS = 0.1;
        epoch = 0;
        switch (j)
        {
        case 0:
            startVal = 0.001;
            break;
        case 1:
            startVal = 0.0003;
            break;
        case 2:
            startVal = -0.05;
            break;
        case 3:
            startVal = 0.005;
            break;
        case 4:
            startVal = -0.073;
        default:
            break;
        }
        for(int i =0;i<new_height*new_width;i++){

            w[i]=startVal;
        }
        w[new_height*new_width] = 1;

        
        start_time = clock();
        while (epoch<max_iterNum && loss>0.032 ){
            if(epoch>5){
                stopValue = w[new_height*new_width];
                printf("%lf\n",stopValue);
            }
            loss = gradient_descent(data,w,data_index,new_height,stepS,&trainA);
            lossptr[epoch] = loss;
            
            end_time = clock();
            time = ((double)(end_time-start_time))/CLOCKS_PER_SEC;
            timeptr[epoch] = time;

            printf("\n%d\n",epoch);
            totalAccurate = 0;
            
            for(int i= 0; i<test_index; i++){

                totalAccurate += testFonc(test,w,new_height,test_index,i);

            }
            testptr[epoch] = (totalAccurate/test_index);
            
            trainptr[epoch] = trainA;
            for(int i = 0; i<new_height*new_width+1; i++){
                wVal[epoch][i] = w[i];
            }
            epoch++;
            printf("\n %d. epoch  %lf  loss  %lf train Accurate %lf time total accurate = %lf\n" , epoch , loss , trainA, time ,(totalAccurate/test_index));
            
            
            
           

        }
        sprintf(output_file, "%sGD%d.txt" ,txt_fileName,j); 
        
        saveResultsToCSV(output_file,1,startVal,lossptr,trainptr,testptr,timeptr,epoch,j,wVal);
    }
    
    for(int i= 0; i<test_index; i++){
       totalAccurate += testFonc(test,w,new_height,test_index,i);
    }

    printf("\n total accurate = %lf\n" ,totalAccurate/test_index);

     printf("\n%lf\n", data[0][new_height*new_width]);

        for(int i =0;i<new_height*new_width;i++){
        w[i]=0.01;
    }
    w[new_height*new_width] = 1;
    /*
    *BU KISIMIN ALTINDA KALANLAR SDG ICIN
    */
    
    // for(int j = 0 ; j<5; j++){
        
    //     loss =100;
    //     stepS = 0.002;
    //     epoch = 0;
    //     switch (j)
    //     {
    //     case 0:
    //         startVal = 0.001;
    //         break;
    //     case 1:
    //         startVal = 0.03;
    //         break;
    //     case 2:
    //         startVal = -0.05;
    //         break;
    //     case 3:
    //         startVal = 0.005;
    //         break;
    //     case 4:
    //         startVal = -0.073;
    //     default:
    //         break;
    //     }
    //     for(int i =0;i<new_height*new_width;i++){

    //         w[i]=startVal;
    //     }
    //     w[new_height*new_width] = 1;

        
    //     start_time = clock();
    //     while (epoch<max_iterNum && loss>0.01 ){
    //         if(epoch>5){
    //             stopValue = w[new_height*new_width];
    //             printf("%lf\n",stopValue);
    //         }
            // loss = Stochastic_Gradient(data,w,data_index,new_height,stepS,&trainA);
    //         lossptr[epoch] = loss;
            
    //         end_time = clock();
    //         time = ((double)(end_time-start_time))/CLOCKS_PER_SEC;
    //         timeptr[epoch] = time;
            
    //         totalAccurate = 0;
            
    //         for(int i= 0; i<test_index; i++){

    //             totalAccurate += testFonc(test,w,new_height,test_index,i);

    //         }
    //         testptr[epoch] = (totalAccurate/test_index);
            
    //         trainptr[epoch] = trainA;
    //         epoch++;
    //         printf("\n %d. epoch  %lf  loss  %lf train Accurate %lf time total accurate = %lf\n" , epoch , loss , trainA, time ,(totalAccurate/test_index));        

    //     }
    //     sprintf(output_file, "%sSGD%d.txt", txt_fileName,j); 
    //     saveResultsToCSV(output_file,2,startVal,lossptr,trainptr,testptr,timeptr,epoch,j,wVal);
    // }
    double *m = (double*)malloc((new_height*new_width*2)*sizeof(double));
    double *v = (double*)malloc((new_height*new_width*2)*sizeof(double));
    double *mVar= (double*)malloc((new_height*new_width*2)*sizeof(double));
    double *vVar = (double*)malloc((new_height*new_width*2)*sizeof(double));
    /*
    * BURASI ADAM ICIN 
    */
    // for(int j = 0 ; j<5; j++){
    //     ;
    //     loss= 10.0;
    //     epoch =0;
    //     stepS = 0.0005;
    //     switch (j)
    //     {
    //     case 0:
    //         startVal = 0.001;
    //         break;
    //     case 1:
    //         startVal = 0.03;
    //         break;
    //     case 2:
    //         startVal = -0.05;
    //         break;
    //     case 3:
    //         startVal = 0.005;
    //         break;
    //     case 4:
    //         startVal = -0.073;
    //     default:
    //         break;
    //     }
    //     for(int i =0;i<new_height*new_width;i++){

    //         w[i]=startVal;
    //     }
    //     w[new_height*new_width] = 1;

    //     start_time = clock();
    //     while (loss>0.03 && epoch<100)
    //     {   
            
       //     loss = adam(data,w,m,v,mVar,vVar,data_index,new_height,stepS , epoch,&trainA);
    //         lossptr[epoch] = loss;

    //         end_time = clock();
    //         time = ((double)(end_time-start_time))/CLOCKS_PER_SEC;
    //         timeptr[epoch] = time;
    //         totalAccurate =0.0;
    //         for(int i= 0; i<test_index; i++){

    //             totalAccurate += testFonc(test,w,new_height,test_index,i);

    //         }
    //         testptr[epoch] = (totalAccurate/test_index);
            
    //         trainptr[epoch] = trainA;
    //         epoch++;
    //         printf(" %d. epoch  %lf  loss  %lf train Accurate %lf time total accurate = %lf\n" , epoch , loss , trainA, time ,(totalAccurate/test_index));
            
            
    //     }
    //     sprintf(output_file, "%sADAM%d.txt", txt_fileName,j); 
    //     saveResultsToCSV(output_file,3,startVal,lossptr,trainptr,testptr,timeptr,epoch,j,wVal);
    // }
    //  printf("%d\n", counter);
           
    //  totalAccurate = 0;
    // for(int i= 0; i<test_index; i++){
    //    totalAccurate += testFonc(test,w,new_height,test_index,i);
    // }

    // printf("\n total accurate = %lf\n" ,totalAccurate/test_index);
    
    for (int i = 0; i < test_index; i++)
    {
        free(test[i]);
    }
    
    for(int i = 0;i<data_index;i++){
        free(data[i]);
    }
    free(timeptr);
    free(lossptr);
    free(trainptr);
    free(testptr);
    free(test);
    free(data);
    free(w);
    free(m);
    free(v);
    free(mVar);
    free(vVar);
    return 0;
}
