build:
	docker build -t mymodelapp .

run:
	docker run -p 5000:5000 mymodelapp
