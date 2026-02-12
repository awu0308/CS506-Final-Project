# CS506-Final-Project

## Description

This project aims to build a system that predicts how many competitive seasons an NBA player has remaining in their career. Using historical player statistics and advanced metrics, the project will estimate a player's remaining competitive window, defined as seasons where they maintain meaningful on-court contributions (e.g. averaging 15+ minutes per game with a player efficency rating above 10).

## Project Goals

Develop a model that predicts the number of remaining competitive seasons for an NBA player given their current age, statistical profile, and career history.

- Build a comprehensive dataset of player-season records labeled with the number of competitive seasons remaining after that point
- Train a model that predicts remaining competitive years within ±1.5 seasons of the true value for at least X% of test cases
- Identify and analyze the most influential features contributing to career longevity (e.g., cumulative minutes, injury history, position played)

## Data Collection

- Pulling structured data through the nba_api Python package
- Supplementing with scraped data from Basketball Reference if needed (e.g. injury history)

## Project Timeline

### Week 1-2: Data Collection and Cleaning
- Collect player-season data using the nba_api Python package and Basketball Reference
- Gather stats going back 20+ seasons to ensure sufficient sample size
- Clean and merge per-game stats, advanced metrics, and injury/games-missed records
- Define and label the target variable (remaining competitive seasons) for each player-season entry

### Week 3-4: Exploratory Data Analysis and Feature Engineering
- Analyze career trajectory patterns across different positions, eras, and draft classes
- Engineer features such as cumulative career minutes, decline trajectory over recent seasons, peak distance, and injury risk
- Visualize relationships between key features and career longevity (e.g., age vs. remaining years by position)

### Week 5-6: Model Development and Training
- Test multiple modeling approaches (e.g., Random Forest, XGBoost, survival analysis)
- Compare regression framing (predicting exact years remaining) vs. classification framing (bucketed categories like 0-1, 2-3, 4-5, 6+ years)
- Evaluate model performance using cross-validation

### Week 7: Model Refinement and Visualization
- Create visualizations including predicted vs. actual remaining years for retired players, survival curves by position, and feature importance charts
- Apply the model to current active players to generate predictions

### Week 8: Final Report

- Finalize everything and possibly prepare to present
