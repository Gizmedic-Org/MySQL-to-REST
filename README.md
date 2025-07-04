# MySQL-to-REST

API automática que expone endpoints RESTful (CRUD y consultas readonly) para todas las tablas y vistas de una base de datos MySQL.
Ideal para exponer tu base como una API moderna con soporte de filtros, selección de campos y ordenamiento tipo OData, sin escribir código adicional.

---

## 🚀 Características principales

- CRUD completo sobre todas las tablas.
- GET-only para todas las vistas.
- Endpoints RESTful limpios y predecibles (`/tabla`, `/tabla/{id}`).
- Soporte de `$filter`, `$select`, `$orderby`, `$page`, `$pageSize` en GET.
- **Integración de endpoints externos mediante `externals.json`** (archivos JSON locales o proxy a APIs externas, sin tocar el código).
- Auto-documentación vía Swagger en `/swagger`.
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

```sh
git clone https://github.com/Gizmedic-Org/MySQL-to-REST.git
cd MySQL-to-REST
```

Configura tu conexión a MySQL creando un archivo .env en el mismo directorio que docker-compose.yml con tu cadena de conexión, el puerto en el que estará expuesta la API
y si los endpoints utilizarán JWT para autenticarse:

<pre>
DATABASE_URL="mysql+pymysql://usuario:contraseña@host:puerto/nombre_base"
API_PORT=8055
JWT_AUTH=true
JWT_SECRET=mysupersecretkeymysupersecretkey
JWT_ALGO=HS256
JWT_USER=admin
JWT_PASS=clave
</pre>

➕ (Opcional) Definición de endpoints externos con externals.json
Puedes agregar endpoints personalizados sin tocar el código, simplemente creando un archivo externals.json junto al main.py (y montando los archivos necesarios si usás Docker).

Ejemplo de externals.json:

<pre>
{
  "externals": [
    {
      "route": "/external/clientes/schema",
      "type": "FILE",
      "destination": "files/clientes_schema.json"
    },
    {
      "route": "/external/objetos",
      "type": "GET",
      "destination": "https://api.restful-api.dev/objects?id=3&id=5"
    }
  ]
}
</pre>

- type "FILE": expone un archivo local como endpoint GET (útil para schemas, catálogos, etc).
- type "GET", "POST", "PUT", "DELETE": expone un proxy al endpoint HTTP especificado, pasando los headers, query y body tal como los recibe tu API.

En Docker, monta el archivo y los directorios que uses:

<pre>
volumes:
  - ./externals.json:/app/externals.json:ro
  - ./files:/app/files:ro
</pre>

Arrancá el contenedor:

```sh
docker compose up --build
```

La API estará disponible en http://localhost:8055 (o el puerto que definas en .env).

## 🧪 Cómo usar
Accede a http://localhost:8055/swagger para ver y probar todos los endpoints generados automáticamente, incluyendo los agregados en externals.json.

### Ejemplo de uso:

GET (con filtros, orden y paginación):

<pre>
GET /usuarios?$filter=activo eq 1&$orderby=nombre desc&$select=id,nombre,apellido&$page=1&$pageSize=10
</pre>

GET un registro por PK:

<pre>
GET /usuarios/1424
</pre>

POST (crear un usuario):

<pre>
POST /usuarios
Content-Type: application/json
{
  "codigoInterno": "test",
  "nombre": "TEST",
  ...
}
</pre>

PUT (modificar un usuario):

<pre>
PUT /usuarios/1424
Content-Type: application/json
{
  "nombre": "TEST ACTUALIZADO",
  "activo": 0
}
</pre>

DELETE (eliminar un usuario):

<pre>
DELETE /usuarios/1424
</pre>

GET externo desde archivo JSON:

<pre>
GET /external/clientes/schema
</pre>

GET proxy a API externa:

<pre>
GET /external/objetos
</pre>

## 📝 Detalles de endpoints
Todos los endpoints devuelven objetos JSON con los siguientes campos:

status: true si la operación fue exitosa, false si hubo un error.

value: datos resultantes o mensaje de error.

En GET: page, pageSize, totalCount para paginación.

El endpoint /swagger expone la documentación OpenAPI.

## ⚠️ Notas importantes
La API detecta automáticamente la estructura de la base (tablas, vistas y PKs).

Los campos ON UPDATE CURRENT_TIMESTAMP y DEFAULT CURRENT_TIMESTAMP se gestionan según las mejores prácticas de MySQL.

Los endpoints definidos en externals.json se agregan automáticamente a la documentación y pueden ser archivos locales o proxys HTTP.

## 💬 Soporte y contribuciones
Pull requests y sugerencias son bienvenidas.
Para soporte, abrí un issue o escribime.

## 🏁 Licencia
MIT License.
Hecho para acelerar tu backend.