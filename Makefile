clean:
	rm audio/*
	rm split/music/*
	rm split/vocal/*

auto:
	python3 auto.py 

upload:
	rclone bisync split/ split:split --verbose

run:
	# python3 auto.py && rclone sync split/ lucifer_drive:split --verbose
	# python3 download.py
	python3 main.py

model:
	wget https://github.com/tsurumeso/vocal-remover/releases/download/v6.0.0b4/vocal-remover-v6.0.0b4.zip && unzip vocal-remover-v6.0.0b4.zip && rm vocal-remover-v6.0.0b4.zip
