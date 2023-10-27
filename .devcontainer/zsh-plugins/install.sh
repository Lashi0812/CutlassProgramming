#!/usr/bin/env bash
set -x
PLUGINS=${PLUGINS:-""}
THEME_URL=${THEME:-""}
OMZSH_PLUGINS=${OMZPLUGINS:-""}
USERNAME=${USERNAME:-$_REMOTE_USER}


# Checks if packages are installed and installs them if not
check_packages() {
	if ! dpkg -s "$@" >/dev/null 2>&1; then
		if [ "$(find /var/lib/apt/lists/* | wc -l)" = "0" ]; then
			echo "Running apt-get update..."
			apt-get update -y
		fi
		apt-get -y install --no-install-recommends "$@"
	fi
}


check_packages git ca-certificates

# ensure zsh is installed
if ! type zsh >/dev/null 2>&1; then
    check_packages zsh
fi 

if [ "$USERNAME" = "root" ]; then
  USER_LOCATION="/root"
else
  USER_LOCATION="/home/$USERNAME"
fi

# ensure oh-my-zsh installed
if ! [ -d $USER_LOCATION/.oh-my-zsh ]; then
  check_packages wget
  sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
  sh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
fi

ZSH_CONFIG="$USER_LOCATION/.zshrc"
THEME_LOCATION="$USER_LOCATION/.oh-my-zsh/custom/themes"
OMZSH_PLUGINS_LOCATION="$USER_LOCATION/.oh-my-zsh/custom/plugins"

# Install custom oh-my-zsh plugins from OMZSH_PLUGINS
currdir=$(pwd)
mkdir -p "$OMZSH_PLUGINS_LOCATION"
cd "$OMZSH_PLUGINS_LOCATION" || exit


IFS=' ' read -ra plugins <<< "${OMZSH_PLUGINS}"

for plugin in "${plugins[@]}"
do
  git clone --depth 1 $plugin
done

cd "$currdir" || exit

mkdir -p "$THEME_LOCATION"
cd "$THEME_LOCATION" || exit

IFS=' ' read -ra themes <<< "${THEME_URL}"

for theme in "${themes[@]}"
do
  git clone --depth 1 $theme
done

cd "$currdir" || exit


# create configuration file if not exists
if ! [ -f "$ZSH_CONFIG" ]; then
  mkdir -p "$(dirname "$ZSH_CONFIG")" && touch "$ZSH_CONFIG"
fi 

# Activate zsh plugins from PLUGINS

sed -i '/^ZSH_THEME/c\ZSH_THEME="powerlevel10k/powerlevel10k"' "$ZSH_CONFIG"
sed -i '1i\
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then\
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"\
fi' "$ZSH_CONFIG" 
sed -i '$a\
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh' "$ZSH_CONFIG"

sed -i -e "s/plugins=.*/plugins=(git ${PLUGINS})/g" "$ZSH_CONFIG"
if  [ -f "$ZSH_CONFIG".pre-oh-my-zsh ]; then
   cat "$ZSH_CONFIG".pre-oh-my-zsh >> "$ZSH_CONFIG"
fi

# check for the conda and init zsh
if [ -f /root/miniconda3/bin/conda ] ; then 
    /root/miniconda3/bin/conda init zsh
fi