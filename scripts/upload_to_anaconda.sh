#!/bin/bash

if [ ! -f scripts/upload_to_anaconda.sh -o ! -f setup.py -o ! -f conda/meta.yaml.in ]; then
	echo "Please run from top dir of repository" >&2
	exit 1
fi
if [ -z "$ANACONDA_TOKEN" ]; then
	echo "Skipping build ... ANACONDA_TOKEN not set." >&2
	exit 1
fi

GATOR_VERSION=$(< setup.cfg awk '/current_version/ {print; exit}' | egrep -o "[0-9.]+")
GATOR_TAG=$(git tag --points-at $(git rev-parse HEAD))
if [[ "$GATOR_TAG" =~ ^v([0-9.]+)$ ]]; then
	LABEL=main
else
	LABEL=dev
	GATOR_VERSION="${GATOR_VERSION}.dev"
	GATOR_TAG=$(git rev-parse HEAD)
fi

echo -e "\n#"
echo "#-- Deploying tag/commit '$GATOR_TAG' (version $GATOR_VERSION) to label '$LABEL'"
echo -e "#\n"

set -eu

< conda/meta.yaml.in sed "s/@GATOR_VERSION@/$GATOR_VERSION/g;" > conda/meta.yaml

# Running build and deployment
# TODO: change gator label to main once in production
conda build conda -c gator/label/dev -c defaults -c conda-forge --user gator --token $ANACONDA_TOKEN --label $LABEL
