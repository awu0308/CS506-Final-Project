.PHONY: all data visualizations model clean

all: data visualizations model

data:
	python data_collection.py

visualizations: nba_career_longevity_data.csv
	python visualizations.py

model: nba_career_longevity_data.csv
	python modeling.py

clean:
	rm -f nba_career_longevity_data.csv
	rm -rf plots/
