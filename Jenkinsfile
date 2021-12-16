
// https://www.jenkins.io/doc/book/pipeline/

node{

    def commit_id 
    def customImage
    def ml_type

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
        ml_type = 'CLASSICAL'
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

            customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./classical-reg/")  
            
        }
    
    stage('Run'){

            if (ml_type == 'CLASSICAL') {
                echo 'I only execute on the master branch'
                customImage.inside {
                    sh 'ls'
                    sh 'echo Hello Classical Regression'
                    // sh 'python3 ./classical-reg/load_data.py'
                    // sh 'python3 ./classical-reg/lr_rf_svr.py'
            }
            
            } 
            else {
                echo 'I execute elsewhere'
            }
                
        
    }
   
    
}
