#!/bin/bash

USER=${LOCAL_USER:-"root"}

if [[ "${USER}" != "root" ]]; then
    USER_ID=${LOCAL_USER_ID:-9001}
    echo ${USER}
    echo ${USER_ID}

    chown ${USER_ID} /home/${USER}
    useradd --shell /bin/bash -u ${USER_ID} -o -c "" -m ${USER}
    usermod -a -G root ${USER}
    adduser ${USER} sudo

    # user:password
    echo "${USER}:123" | chpasswd

    export HOME=/home/${USER}
    export PATH=/home/${USER}/.local/bin/:$PATH
else
    export PATH=/root/.local/bin/:$PATH
fi

cd $HOME
exec gosu ${USER} "$@"