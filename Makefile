.PHONY: create-env install-dependencies download-data-wind generate-modis-data start-app clean-cache train eval all

## INFO: Colors Mapping for Cleaner Logging 


RESET := \033[0m

GREEN := \033[1;32m
RED := \033[1;31m
YELLOW := \033[1;33m
BLUE := \033[1;34m

## INFO: Logging Essentials

INFO := $(GREEN)[info]$(RESET):\n\t
WARNING := $(YELLOW)[warning]$(RESET):\n\t 
ERROR := $(RED)[error]$(RESET):\n\t
DECISION := $(BLUE)[decision]$(RESET):\n\t

## INFO: FILENAMES (CHANGE WHENEVER YOU LIKE, Make Sure the files have the same structure for arguments to work)

TRAINING_FILE := src.scripts.training
EVALUATION_FILE := src.scripts.evaluate

## INFO: Default Settings

PYTHON_VERSION ?= 3.12.11 # currently compatible version
ENV ?= fire_env
FILENAME ?= CONV_LSTM
VERBOSE ?= FALSE
LOG_LOSS ?= TRUE
EPOCHS ?= 10
WEIGHT_DECAY ?= 0.0005
LR ?= 1e-3
PATIENCE ?= 3
HELP ?= False
N_TIME=1d


MODIS_GENERATE_FILE := src.data_scripts.make_dataset_script
MODIS_YEARS ?= []
MODIS_BATCH ?= 10
MODIS_MODE ?= 0
MODIS_BIN ?= 0.5


create-env:
	conda create -n $(ENV) python=$(PYTHON_VERSION) || {
		@printf "$(ERROR)Failed to Create Environment!\n";\
		exit 1;\
	}
	@printf "$(INFO)Conda environment created under the name $(GREEN)$(ENV)$(RESET)\n\tPython version: $(PYTHON_VERSION)\n"

install-dependencies:
	@printf "$(INFO)Installing Requirements:\n"
	@pip install -r requirements.txt || { \
		@printf "$(ERROR)Failed to Install Dependencies!\n";\
		exit 1;\
	}
	@printf "$(INFO)Installation Process Completed"
	
download-data:
	@printf "$(ERROR)$(RED)Not Implemented Yet.$(RESET) Soon...\n" 

start-app:
	@printf "$(ERROR)$(RED)Work In Process...$(RESET)\n"

clean-cache:
	@printf "$(INFO)Cleaning Cache...\n"
	@python -m src.scripts.clean_cache || {\
		@printf "$(WARNING) Some Files Counldn't Be Cleaned!\n";\
		exit 1;\
	}
	@printf "$(INFO)Cleaning Complete.\n"

train:
	@printf "$(INFO)TRAINING OPTIONS:\n\t\t$(GREEN)Epochs$(RESET): $(EPOCHS)\n\t\t$(GREEN)MODEL NAME$(RESET): $(FILENAME)\n\t\t$(GREEN)Training Log Shown$(RESET): $(VERBOSE)\n\t\t$(GREEN)Save Training Loss$(RESET)?: $(LOG_LOSS)\n"
	@printf "\t\t$(GREEN)Learning Rate$(RESET): $(LR)\n\t\t$(GREEN)Weight Decay$(RESET): $(WEIGHT_DECAY)\n\t\t$(GREEN)Scheduler Patience$(RESET): $(PATIENCE)\n"
	@bash -c '\
		printf "$(DECISION)Do you want to proceed with the training? (y|$(BLUE)N$(RESET)): ";\
		read ans; \
		if [ "$$ans" != "y" ] && [ "$$ans" != "Y" ]; then \
			printf "$(WARNING)Training Aborted by the user.\n";\
			exit 1;\
		fi; \
		printf "$(INFO)Beginning Training\n"; \
		python -m $(TRAINING_FILE) --epochs $(EPOCHS) --filename $(FILENAME) --verbose $(VERBOSE) --log-loss $(LOG_LOSS) --lr $(LR) --weight-decay $(WEIGHT_DECAY) --scheduler-patience $(PATIENCE) || {\
			printf "$(ERROR)FATAL Error Faced During Training. Aborting...\n";\
			exit 1;\
		}; \
		printf "$(INFO)Training Complete.\n" \
	'
train-script:
	python $(TRAINING_SCRIPT) || {\
		printf "$(ERROR)FATAL Error Faced During Training Using Script. Aborting...\n";\
		exit 1;\
	};\
	printf "$(INFO)Training Completed Using Script.\n"

generate-modis-data:
	@python -m $(MODIS_GENERATE_FILE) --years $(MODIS_YEARS) --batch_size $(MODIS_BATCH) --mode $(MODIS_MODE) --time $(N_TIME) --bin_size $(MODIS_BIN)


eval:
	@printf "$(INFO)EVALUATION OF MODEL:"
	python -m $(EVALUATION_FILE)



full-process: install-dependencies clean-cache train
