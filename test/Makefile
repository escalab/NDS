PROJECTS ?= $(shell find . -mindepth 2 -maxdepth 2 -name Makefile)
FILTER_OUT := 
PROJECTS := $(filter-out $(FILTER_OUT),$(PROJECTS))

%.ph_build :
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	+@$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)
	
all:  $(addsuffix .ph_build,$(PROJECTS))

build: $(addsuffix .ph_build,$(PROJECTS))
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
