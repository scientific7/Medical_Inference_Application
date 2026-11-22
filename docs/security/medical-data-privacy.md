# Medical data privacy

Use the platform for research under the applicable data-handling policy.
The inference datastore retains prediction metadata and successful-input
checksums. These can be sensitive. Protect database access, backups, and
retention settings accordingly. The service does not persist original image
bytes in its datastore. Multipart parsing can use transient files on the
process temp filesystem, and queued batches hold images in memory.
Deploy with a protected temp mount and restrict infrastructure operator access.

Streamlit keeps uploads in session memory and uses them to render overlays.
The dashboard can export resulting metadata. Clear shared workstation sessions
after use. This platform has no clinical or compliance certification.
