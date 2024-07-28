#!/bin/bash

cp ../params.cfg .

# Run the visualization
python3 -m http.server &
firefox http://localhost:8000/visualization.html