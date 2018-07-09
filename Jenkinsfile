#!groovy

// Jenkinsfile for compiling, testing, and packaging MMG


pipeline {
    agent {
        dockerfile true
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                sh 'mkdir -p build'
            }
        }
        stage('Compile') {
            steps {
                sh '''
                	cd build &&
                	cmake -D CMAKE_BUILD_TYPE=Debug -D BUILD_TESTS=ON .. &&
                	make
                '''
            }
        }
        stage('Test') {
            steps {
                sh '''
                	cd build &&
                  echo "TODO: run tests"
                '''
            }
        }
}
