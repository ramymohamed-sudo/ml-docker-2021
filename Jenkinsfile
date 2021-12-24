
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
        // ml_type = 'RNN'
        ml_type = 'PULL_KERAS'

    }


    stage('Build'){

        // OR build from Dockerfile  // from Dockerfile in "./"
        // customImage = docker.build("my-image:${env.BUILD_ID}", "./") 
        if (ml_type == 'CLASSICAL') {
            customImage = docker.build("ramyrr/machinelearning_sklearn:${commit_id}", "./classical-reg/") 
        }
        else if (ml_type == 'CNN') {
            echo 'The CNN model is is use'
            customImage = docker.build("ramyrr/machinelearning_keras:${commit_id}", "./classical-reg/") 
        }
        else if (ml_type == 'RNN') {
            echo 'The RNN model is is use'
            customImage = docker.build("ramyrr/machinelearning_keras:${commit_id}", "./rnn/")  
        }
        else if (ml_type == 'PULL_KERAS') {
            // Build from image        
            def myTestContainer = docker.image('ramyrr/machinelearning_keras:latest')
            myTestContainer.pull()
            myTestContainer.inside{
                    sh 'ls'
                    sh 'echo Hello RNN-based Regression inside the docker'
                    sh 'python3 ./rnn/load_data_4_files_1D_2D.py'
                    sh 'python3 ./rnn/rnn_one_layer_2D.py'
            }
        }

        else {
            echo 'default case'
        }

        }
    

    stage('TEST'){                  // TEST FIREFOX and EDGE

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
                    sh 'python3 ./rnn/load_data_4_files_1D_2D.py'
                    sh 'python3 ./rnn/rnn_one_layer_2D.py'

            }
            }

            else {
                echo 'default case'
            }
                
        
    }


    stage('STAGING'){
        echo 'STAGING stage in Jenkins'
    }


    // stage('Push'){
    //     echo 'PUSH stage in Jenkins'
    //     docker.withRegistry('https://index.docker.io/v1/', '7ec5aa2d-ed10-4282-ba0a-527c27a55a11'){  
    //         // 'dockerhub'   replaced with '7ec5aa2d-ed10-4282-ba0a-527c27a55a11'
    //         // def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()            
    //         customImage.push()
    //     }
    // }  

    stage('PRODUCTION'){
        echo 'PRODUCTION stage in Jenkins'
    }
   
    
}
