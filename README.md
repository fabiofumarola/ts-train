# Descrizione repository

Per la corretta gestione della Software Development LifeCycle e per permettere l’utilizzo dell’automazione, il repository di tipo Python deve presentare la seguente struttura nel branch develop:

* *pipeline.yaml*: contiene puntamenti ai repository Nexus; non deve essere modificato se si tratta di common library, mentre se si tratta di una project library andrà inserito il nome del repository Nexus di progetto <br/><br/>
* *cartella .alm*: contiene il file build.jenkinsFile utilizzato per l'implementazione della CI/CD in ambiente di sviluppo <br/><br/>
* *pyproject.toml*: deve essere personalizzato dal team di sviluppo con i dati che riguardano la libreria che si sta sviluppando <br/><br/>

# Step per buildare la libreria python in sviluppo

### 1. Applicare una branching strategy

La gestione della strategia di “branch” (e di conseguenza di “merge”) è sicuramente uno degli elementi più delicati dello sviluppo del software. Per questo motivo, una volta definita, la strategia di branch deve essere seguita in maniera precisa e senza eccezioni, sia per permettere una lettura “storica” del software sia perché questa abilita l’automazione di alcune steps tramite Jenkins.

Al seguente [link](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/branching/branching-strategy.md) è possibile consultare la guida alla Branching Strategy proposta in Poste Italiane.

E' possibile naturalmente definirne una diversa che deve essere documentata nel repository, in modo da poterla consultare anche successivamente.

### 2. Personalizzare i file di configurazione

Come descritto nelle [linee guida](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/svil-python-common.md), prima di proseguire con la definizione del job su jenkins e la relativa esecuzione, andranno personalizzati il file "pyproject.toml" inserendo i dati relativi alla libreria che si sta sviluppando ed, eventualmente, il file build.jenkinsFile se si vuole eseguire la build partendo da un branch diverso da develop.

### 3. Verificare i prerequisti per poter utilizzare Jenkins in ambiente di sviluppo

Per poter procedere con il processo automatico di build e deploy, occorre verificare che i seguenti [prerequisiti](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/prerequisiti.md) siano soddisfatti.

### 4. Creare il job di build su Jenkins

Al seguente [link](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/svil-python-common.md) è possibile consultare il manuale operativo per la creazione del job jenkins che esegue la build della libreria.

### 5. Eseguire i job su jenkins

In base al [jenkins di sviluppo](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/prerequisiti.md#accesso-jenkins-disponibili-in-ambiente-di-sviluppo-tramite-vpn) usato, collegarsi alla console ed eseguire il job di build.

# Utilizzo libreria buildata in ambiente di sviluppo

Per utilizzare queste librerie nei propri progetti pyhton, è sufficiente aggiungere nel requirement.txt (utilizzato da pip) le seguenti righe:

- In caso di commoon library:

--extra-index-url https://nexus.alm.poste.it/repository/python-common-library-snapshot/simple <br>
< my package X >==< my package version X > <br>
< my package Y >==< my package version Y > <br>
dove "my package X" è il nome del whl creato e "my package version X" è la versione da usare.


- in caso di project library

--extra-index-url https://nexus.alm.poste.it/repository/< PROJECT REPO NAME >-snapshot/simple <br>
< my package X >==< my package version X > <br>
< my package Y >==< my package version Y > <br>
dove "my package X" è il nome del whl creato e "my package version X" è la versione da usare.

# Creazione release della libreria

Per poter poi creare la versione "release" della libreria deve essere creato il job (secondo quanto riportato nella seguente [guida](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/release-python-project.md)) e richiesto successivamente l'export tramite [ticket](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/guida-sviluppo/ticket_promozione_pipeline.md).

# Utilizzo libreria buildata in ambiente di release

Per utilizzare queste librerie nei propri progetti pyhton, è sufficiente aggiungere nel requirement.txt (utilizzato da pip) le seguenti righe:

- In caso di common library:

--extra-index-url https://nexus.alm.poste.it/repository/python-common-library/simple <br>
< my package X >==< my package version X > <br>
< my package Y >==< my package version Y > <br>
dove "my package X" è il nome del whl creato e "my package version X" è la versione da usare.


- in caso di project library

--extra-index-url https://nexus.alm.poste.it/repository/< PROJECT REPO NAME >/simple <br>
< my package X >==< my package version X >  <br>
< my package Y >==< my package version Y > <br>
dove "my package X" è il nome del whl creato e "my package version X" è la versione da usare.

# Link utili

* [Troubleshooting](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/jenkins/troubleshooting.md)
* [Guida all'apertura dei ticket](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/guida-sviluppo/Apertura_Ticket_ALM.md)
* [Template Manuale di installazione](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/guida-sviluppo/linee_guida_manuale_installazione.md)
* [Template Release Notes](https://gitlab.alm.poste.it/guidelines/linee-guida-alm/-/blob/master/guida-sviluppo/template_release_notes.md)


