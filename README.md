# MySQL-to-REST

API automática que expone endpoints RESTful (CRUD y consultas readonly) para todas las tablas y vistas de una base de datos MySQL.
Ideal para exponer tu base como una API moderna con soporte de filtros, selección de campos y ordenamiento tipo OData, sin escribir código adicional.

---

## 🚀 Características principales

- CRUD completo sobre todas las tablas.
- GET-only para todas las vistas.
- Endpoints RESTful limpios y predecibles (/tabla, /tabla/{id}).
- Soporte de $filter, $select, $orderby, $page, $pageSize en GET.
- Auto-documentación vía Swagger en /swagger.
- Respuestas uniformes, ideales para integración directa.
- Sin definición de modelos manual ni configuración extra.

---

## ⚡ Requisitos

- Docker y Docker Compose instalados en tu sistema.
- Base de datos MySQL accesible desde el contenedor.
- Python 3.11+ si deseas correrlo fuera de Docker.

---

## 📂 Instalación y puesta en marcha

Cloná el repositorio (o copiá estos archivos en tu proyecto):

git clone https://github.com/Gizmedic-Org/MySQL-to-REST.git
cd MySQL-to-REST

Configura tu conexión a MySQL creando un archivo .env en el mismo directorio que docker-compose.yml con tu cadena de conexión:

DATABASE_URL="mysql+pymysql://usuario:contraseña@host:puerto/nombre_base"
API_PORT=8055

Arrancá el contenedor:

docker compose up --build

La API estará disponible en http://localhost:8055 (o el puerto que definas en .env).

---

## 🧪 Cómo usar

Accede a http://localhost:8055/swagger para ver y probar todos los endpoints generados automáticamente.

---

Ejemplo de uso:

1. GET (con filtros, orden y paginación):

GET /usuarios?$filter=activo eq 1&$orderby=nombre desc&$select=id,nombre,apellido&$page=1&$pageSize=10

2. GET un registro por PK:

GET /usuarios/1424

3. POST (crear un usuario):

POST /usuarios
Content-Type: application/json

{
  "codigoInterno": "test",
  "nombre": "TEST",
  "apellido": "",
  "numeroCedula": "12345678",
  "numeroCredencial": "12345678",
  "fechaNacimiento": "1966-04-22T00:00:00",
  "genero": "M",
  "whatsapp": "4444444",
  "activo": 1,
  "token": "alekfjsklmwsoiefjflsmdflds"
}

4. PUT (modificar un usuario, con id en la URL):

PUT /usuarios/1424
Content-Type: application/json

{
  "nombre": "TEST ACTUALIZADO",
  "activo": 0
}

5. DELETE (eliminar un usuario):

DELETE /usuarios/1424

---

## 📝 Detalles de endpoints

- Todos los endpoints devuelven objetos JSON con los siguientes campos:
  - status: true si la operación fue exitosa, false si hubo un error.
  - value: datos resultantes o mensaje de error.
  - En GET: page, pageSize, totalCount para paginación.
- El endpoint /swagger expone la documentación OpenAPI.

---

## ⚠️ Notas importantes

- La API detecta automáticamente la estructura de la base (tablas, vistas y PKs).
- Los campos ON UPDATE CURRENT_TIMESTAMP y DEFAULT CURRENT_TIMESTAMP se gestionan según las mejores prácticas de MySQL.

---

## 💬 Soporte y contribuciones

Pull requests y sugerencias son bienvenidas.
Para soporte, abrí un issue o escribime.

---

## 🏁 Licencia

MIT License.
Hecho para acelerar tu backend.
