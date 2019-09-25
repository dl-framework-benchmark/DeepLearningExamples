for HOST in 172.31.24.16 172.31.29.249 172.31.26.206 172.31.16.106
do
	rsync -a --exclude /home/ubuntu/DeepLearningExamples/results /home/ubuntu/DeepLearningExamples \
    $HOST:~
done
