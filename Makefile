PYTHON ?= python

.PHONY: help check-structure checklist-e1

help:
	@echo "Comandos disponibles:"
	@echo "  make check-structure  - Verifica carpetas base del proyecto"
	@echo "  make checklist-e1     - Muestra checklist rapido de Semana 2"

check-structure:
	@echo "Verificando estructura minima..."
	@$(PYTHON) -c "from pathlib import Path; req=['docs/PROJECT_DOCUMENTATION.md','README.md','.env.example','.gitignore','requirements.txt','src/api','src/core','src/rag','src/security','tests']; missing=[p for p in req if not Path(p).exists()]; print('OK: estructura completa' if not missing else 'FALTAN: ' + ', '.join(missing)); exit(0 if not missing else 1)"

checklist-e1:
	@echo "[ ] Caso de uso definido"
	@echo "[ ] Tabla IN SCOPE / OUT OF SCOPE completa"
	@echo "[ ] Minimo 5 RF con criterios medibles"
	@echo "[ ] Minimo 4 RNF con umbrales"
	@echo "[ ] Restricciones y supuestos documentados"
	@echo "[ ] Stack preliminar definido"
