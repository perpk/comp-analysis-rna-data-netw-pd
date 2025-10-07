install: requirements.txt
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

clean:
	rm -rf src/__pycache__