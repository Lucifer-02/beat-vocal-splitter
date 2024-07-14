clean:
	rm audio/*
	rm split/music/*
	rm split/vocal/*

auto:
	python3 auto.py 

upload:
	rclone sync split/ lucifer_drive:split --verbose

run:
	python3 auto.py && rclone sync split/ lucifer_drive:split --verbose
