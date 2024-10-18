# Deep Reinforcement Learning from Human Preferences
DRLHP/rl-teacher rewritten in PyTorch 2.4 and Django 5.0.x

Original repo: https://github.com/mrahtz/learning-from-human-preferences
Reference repos: https://github.com/nottombrown/rl-teacher, https://github.com/HumanCompatibleAI/learning-from-human-preferences


#### Set up the `human-feedback-api` webapp
First you'll need to set up django. This will create a `db.sqlite3` in your local directory.

    python src/manage.py migrate
    python src/manage.py collectstatic

Start the webapp

    python src/manage.py runserver 0.0.0.0:8000


### Project Status:

Currently still a work in progress. Planning on refactor to UI similar to the HumanCompatibleAI impl and compatible with Torchrl. Using an updated version of the webapp from rl-teacher.
