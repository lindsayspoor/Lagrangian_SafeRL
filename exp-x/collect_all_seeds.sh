#!/bin/bash

# Get all directories with 'fixed_lambda_'
LAMBDA_DIRS=$(find . -maxdepth 1 -type d -name "*fixed_lambda_*" | sed 's|^\./||')

for EXP_DIR in $LAMBDA_DIRS; do
    echo "Processing experiment directory: $EXP_DIR"

    # Normalize output directory (remove timestamp suffix)
    OUTPUT_DIR=$(echo "$EXP_DIR" | sed -E 's/(_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})$//')
    echo "Output dir: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Find all progress.csv files under seed_* folders
    PROGRESS_FILES=$(find "$EXP_DIR" -type f -path "*/seed_*/progress.csv")

    for PROGRESS_PATH in $PROGRESS_FILES; do
        # Extract the seed name (first folder with seed_* in path)
        SEED=$(echo "$PROGRESS_PATH" | grep -oE 'seed_[^/]+')

        if [ -n "$SEED" ]; then
            mkdir -p "$OUTPUT_DIR/$SEED"
            cp "$PROGRESS_PATH" "$OUTPUT_DIR/$SEED/"
            echo "    Copied: $PROGRESS_PATH -> $OUTPUT_DIR/$SEED/"
        else
            echo "    ! Could not determine seed folder from: $PROGRESS_PATH"
        fi
    done
done


# Get all directories with 'auto_update_GA_'
LAMBDA_DIRS=$(find . -maxdepth 1 -type d -name "*auto_update_GA_*" | sed 's|^\./||')

for EXP_DIR in $LAMBDA_DIRS; do
    echo "Processing experiment directory: $EXP_DIR"

    # Normalize output directory (remove timestamp suffix)
    OUTPUT_DIR=$(echo "$EXP_DIR" | sed -E 's/(_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})$//')
    echo "Output will go to: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Find all progress.csv files under seed_* folders
    PROGRESS_FILES=$(find "$EXP_DIR" -type f -path "*/seed_*/progress.csv")

    for PROGRESS_PATH in $PROGRESS_FILES; do
        # Extract the seed name (first folder with seed_* in path)
        SEED=$(echo "$PROGRESS_PATH" | grep -oE 'seed_[^/]+')

        if [ -n "$SEED" ]; then
            mkdir -p "$OUTPUT_DIR/$SEED"
            cp "$PROGRESS_PATH" "$OUTPUT_DIR/$SEED/"
            echo "    Copied: $PROGRESS_PATH -> $OUTPUT_DIR/$SEED/"
        else
            echo "    ! Could not determine seed folder from: $PROGRESS_PATH"
        fi
    done
done


# Get all directories with 'auto_update_GA_'
LAMBDA_DIRS=$(find . -maxdepth 1 -type d -name "*auto_update_PID_*" | sed 's|^\./||')

for EXP_DIR in $LAMBDA_DIRS; do
    echo "Processing experiment directory: $EXP_DIR"

    # Normalize output directory (remove timestamp suffix)
    OUTPUT_DIR=$(echo "$EXP_DIR" | sed -E 's/(_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})$//')
    echo "Output will go to: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Find all progress.csv files under seed_* folders
    PROGRESS_FILES=$(find "$EXP_DIR" -type f -path "*/seed_*/progress.csv")

    for PROGRESS_PATH in $PROGRESS_FILES; do
        # Extract the seed name (first folder with seed_* in path)
        SEED=$(echo "$PROGRESS_PATH" | grep -oE 'seed_[^/]+')

        if [ -n "$SEED" ]; then
            mkdir -p "$OUTPUT_DIR/$SEED"
            cp "$PROGRESS_PATH" "$OUTPUT_DIR/$SEED/"
            echo "    Copied: $PROGRESS_PATH -> $OUTPUT_DIR/$SEED/"
        else
            echo "    ! Could not determine seed folder from: $PROGRESS_PATH"
        fi
    done
done


echo "Done."
