#!groovy

pipeline {
    agent any
    environment {
        EB_CUSTOM_REPOSITORY = '/users/simonpi/jenkins/production/easybuild'
    }
    stages {
        stage('Checkout') {
            steps {
                node('ssl_daintvm1') {
                    dir('SIRIUS') {
                        checkout scm
                        echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                    }
                }
            }
        }
        stage('Compile') {
            steps {
                node('ssl_daintvm1') {
                    dir('SIRIUS') {
                        sh '''
                           #!/bin/bash -l
                           sbatch --wait ci/build-daint-gpu.sh
                           echo "---------- build-daint-gpu.out ----------"
                           cat build-daint-gpu.out
                           echo "---------- build-daint-gpu.err ----------"
                           cat build-daint-gpu.err
                           '''
                    }
                }
            }
        }
        stage('Test') {
            steps {
                node('ssl_daintvm1') {
                    dir('SIRIUS') {
                        sh '''
                    	     cd build &&
                           echo "TODO: run tests...."
                           '''
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/*.out', fingerprint: true
            archiveArtifacts artifacts: '**/*.err', fingerprint: true
        }
    }
}
