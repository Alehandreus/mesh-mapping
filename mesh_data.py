from dataclasses import dataclass
from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder, GPURayTracer


@dataclass
class PyMesh:
    """Wraps mesh data with GPU helpers for sampling, traversal, and ray tracing."""

    mesh: Mesh
    mesh_split: Mesh
    sampler: GPUMeshSampler
    traverser: GPUTraverser
    ray_tracer: GPURayTracer

    @classmethod
    def from_file(cls, path: str, *, n_samples: int = 100_000, bvh_depth: int = 25) -> "PyMesh":
        mesh = Mesh.from_file(path)

        builder = CPUBuilder(mesh)
        bvh = builder.build_bvh(bvh_depth)
        mesh = Mesh.from_data(bvh.get_vertices(), bvh.get_faces())

        sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, n_samples)
        traverser = GPUTraverser(bvh)
        ray_tracer = GPURayTracer(bvh)

        # Keep a split-able copy around for training encoders that expect more faces.
        mesh_split = Mesh.from_data(bvh.get_vertices(), bvh.get_faces())
        return cls(mesh, mesh_split, sampler, traverser, ray_tracer)


def save_mesh_previews(meshes, size: int = 512) -> None:
    """Save simple previews for a collection of PyMesh objects keyed by name."""
    for name, py_mesh in meshes.items():
        mesh = py_mesh.mesh
        mesh.save_preview(f"{name}_mesh_preview.png", size, size, mesh.get_c(), mesh.get_R())
