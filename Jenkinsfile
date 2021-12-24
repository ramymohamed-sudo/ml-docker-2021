
// https://www.jenkins.io/doc/book/pipeline/

node{

    def commit_id 
    def customImage
    def ml_type

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
        // ml_type = 'CLASSICAL'
        ml_type = 'RNN'

    }


    stage('Build'){

        // Build from image        
        //        def myTestContainer = docker.image('jupyter/scipy-notebook')
        //         myTestContainer.pull()
        //         myTestContainer.inside{
        //              sh 'pip install joblib'
        //              sh 'python3 train.py'
        //         }

        // OR build from Dockerfile  // from Dockerfile in "./"
        // customImage = docker.build("my-image:${env.BUILD_ID}", "./") 
        if (ml_type == 'CLASSICAL') {
            customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./classical-reg/") 
        }
        else if (ml_type == 'CNN') {
            echo 'The CNN model is is use'
            customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./CNN/") 
        }
        else if (ml_type == 'RNN') {
            echo 'The RNN model is is use'
            customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./classical-reg/")  
        }
        else {
            echo 'default case'
        }

        }
    

    stage('Run'){

            if (ml_type == 'CLASSICAL') {
                echo 'I Run the classical ML model'
                customImage.inside {
                    sh 'ls'
                    sh 'echo Hello Classical Regression'
                    // sh 'python3 ./classical-reg/load_data.py'
                    // sh 'python3 ./classical-reg/lr_rf_svr.py'
            }
            } 

            else if (ml_type == 'CNN') {
                echo 'I Run the CNN ML model when it is ready'
            }

            else if (ml_type == 'RNN') {
                echo 'I Run the CNN ML model'
                customImage.inside {
                    sh 'ls'
                    sh 'echo Hello RNN-based Regression'
                    echo 'run from classical folder ??'
                    sh 'python3 ./classical-reg/load_data_4_files_1D_2D.py'
                    sh 'python3 ./classical-reg/ rnn_1layer_2D.py'
                    // sh 'python3 ./classical-reg/load_data.py'
                    // sh 'python3 ./classical-reg/lr_rf_svr.py'
            }
            }

            else {
                echo 'default case'
            }
                
        
    }
   
    
}
