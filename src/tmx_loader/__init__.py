import pathlib
from collections.abc import MutableMapping
from typing import Final

import attrs
import panda3d.core as p3d
import pytiled_parser as ptp
import pytiled_parser.tiled_object as ptp_obj

HORIZONTAL_FLIP_FLAG: Final = 0x80000000
VERTICAL_FLIP_FLAG: Final = 0x40000000
DIAGONAL_FLIP_FLAG: Final = 0x20000000
HEXAGONAL_ROTATION_FLAG: Final = 0x10000000


class UnsupportedError(ValueError):
    pass


def _make_default_tile_vertex_data():
    vertex_format = p3d.GeomVertexFormat.get_v3t2()
    vertex_data = p3d.GeomVertexData('tile', vertex_format, p3d.Geom.UH_static)
    return vertex_data


@attrs.define
class TileGeomBuilder:
    vertex_data: p3d.GeomVertexData = attrs.Factory(_make_default_tile_vertex_data)

    def add_tile(self, shape_xform: p3d.LMatrix4, uv_xform: p3d.LMatrix4) -> None:
        vertex_writer = p3d.GeomVertexWriter(self.vertex_data, 'vertex')
        texcoord_writer = p3d.GeomVertexWriter(self.vertex_data, 'texcoord')
        vertex_writer.set_row(self.vertex_data.get_num_rows())
        texcoord_writer.set_row(self.vertex_data.get_num_rows())
        points = ((0, 0), (0, 1), (1, 1), (1, 1), (1, 0), (0, 0))
        for x, y in points:
            vertex = shape_xform.xform_point((x, 0, y))
            texcoord = uv_xform.xform_point((x, 0, y))
            vertex_writer.add_data3(vertex)
            texcoord_writer.add_data2(texcoord.x, texcoord.z)

    def generate_geom(self) -> p3d.Geom:
        primitive = p3d.GeomTriangles(p3d.Geom.UH_static)
        primitive.add_next_vertices(self.vertex_data.get_num_rows())
        primitive.close_primitive()
        geom = p3d.Geom(self.vertex_data)
        geom.add_primitive(primitive)
        return geom


def extract_tile_transform(gid: int) -> tuple[int, p3d.LMatrix4]:
    transform = p3d.LMatrix4()
    if gid & DIAGONAL_FLIP_FLAG:
        transform *= p3d.LMatrix4.rotate_mat(180, (1, 0, -1))
        transform *= p3d.LMatrix4.translate_mat(1, 0, -1)
    if gid & HORIZONTAL_FLIP_FLAG:
        transform *= p3d.LMatrix4.rotate_mat(180, (0, 0, 1))
        transform *= p3d.LMatrix4.translate_mat(1, 0, 0)
    if gid & VERTICAL_FLIP_FLAG:
        transform *= p3d.LMatrix4.rotate_mat(180, (1, 0, 0))
        transform *= p3d.LMatrix4.translate_mat(0, 0, 1)
    # Clear the transform flags.
    gid &= ~(
        HORIZONTAL_FLIP_FLAG
        | DIAGONAL_FLIP_FLAG
        | VERTICAL_FLIP_FLAG
        | HEXAGONAL_ROTATION_FLAG
    )
    return gid, transform


def get_tile_uv_transform(tileset: ptp.Tileset, tile_id: int) -> p3d.LMatrix4:
    tile_scale = p3d.LMatrix4.scale_mat(tileset.tile_width, 1, tileset.tile_height)
    if tileset.image_height is None or tileset.image_width is None:
        raise ValueError(f"Missing image dimensions for tileset {tileset.name!r}")
    shift = p3d.LMatrix4.translate_mat(
        tileset.margin, 0, tileset.image_height - tileset.margin - tileset.tile_height
    )
    image_scale = p3d.LMatrix4.scale_mat(
        1 / tileset.image_width, 1, 1 / tileset.image_height
    )
    j, i = divmod(tile_id, tileset.columns)
    x_start = i * (tileset.tile_width + tileset.spacing)
    y_start = j * (tileset.tile_height + tileset.spacing)
    translation = p3d.LMatrix4.translate_mat(x_start, 0, -y_start)
    return tile_scale * shift * translation * image_scale


def load_tileset_image(tileset: ptp.Tileset) -> p3d.Texture:
    assert tileset.image is not None
    loader_options = p3d.LoaderOptions()
    loader_options.set_auto_texture_scale(p3d.ATS_none)
    texture = p3d.Texture()
    texture.read(tileset.image, options=loader_options)
    texture.magfilter = p3d.SamplerState.FT_nearest
    return texture


def make_render_state(texture: p3d.Texture) -> p3d.RenderState:
    texture_attrib = p3d.TextureAttrib.make(texture)
    if texture.num_components > 3:
        alpha_attrib = p3d.TransparencyAttrib.make(p3d.TransparencyAttrib.M_alpha)
        render_state = p3d.RenderState.make(texture_attrib, alpha_attrib)
    else:
        render_state = p3d.RenderState.make(texture_attrib)
    return render_state


def modify_opacity(texture: p3d.Texture, opacity: float) -> p3d.Texture:
    image = p3d.PNMImage(texture.x_size, texture.y_size)
    texture.store(image)
    image.alpha_fill(opacity)
    new_texture = p3d.Texture()
    new_texture.load(image)
    return new_texture


def load_image_layer(layer: ptp.ImageLayer) -> p3d.NodePath:
    layer_node = p3d.NodePath(layer.name)
    card_maker = p3d.CardMaker(layer.image.stem)
    texture = p3d.Texture()
    texture.read(layer.image)
    width, height = texture.orig_file_x_size, texture.orig_file_y_size
    texture = modify_opacity(texture, layer.opacity)
    card_maker.set_frame(0, width, -height, 0)
    image_node = p3d.NodePath(card_maker.generate())
    image_node.set_state(make_render_state(texture))
    image_node.reparent_to(layer_node)
    return layer_node


def load_collider(tiled_object: ptp_obj.TiledObject) -> p3d.CollisionSolid:
    if not isinstance(tiled_object, ptp_obj.Rectangle):
        raise UnsupportedError("Non-rectangular colliders not yet supported")
    width, height = tiled_object.size
    collider = p3d.CollisionBox((0, -16, -height), (width, 16, 0))
    return collider


@attrs.define
class TileArranger:
    tileset: ptp.Tileset
    geom_builder: TileGeomBuilder = attrs.Factory(TileGeomBuilder)

    def add_tile(self, gid: int, x: int, y: int) -> None:
        if not gid:
            return
        gid, tile_transform = extract_tile_transform(gid)
        tile_scale = p3d.LMatrix4.scale_mat(
            self.tileset.tile_width, 1, self.tileset.tile_height
        )
        tile_transform *= p3d.LMatrix4.translate_mat(x, 0, -y - 1) * tile_scale
        uv_transform = get_tile_uv_transform(self.tileset, gid - self.tileset.firstgid)
        self.geom_builder.add_tile(tile_transform, uv_transform)

    def generate_node(self) -> p3d.GeomNode:
        node = p3d.GeomNode('tile')
        tile_geom = self.geom_builder.generate_geom()
        texture = load_tileset_image(self.tileset)
        render_state = make_render_state(texture)
        node.add_geom(tile_geom, render_state)
        return node


@attrs.define
class ColliderHandler:
    colliders: MutableMapping[int, p3d.PandaNode] = attrs.Factory(dict)

    def load_colliders(self, tileset: ptp.Tileset) -> None:
        if tileset.tiles is None:
            return
        for tile in tileset.tiles.values():
            if not isinstance(tile.objects, ptp.ObjectLayer):
                continue
            collision_node = p3d.CollisionNode('collision_node')
            for tiled_object in tile.objects.tiled_objects:
                collision_node.add_solid(load_collider(tiled_object))
            self.colliders[tileset.firstgid + tile.id] = collision_node

    def get_collider(self, gid: int) -> p3d.PandaNode | None:
        return self.colliders.get(gid)


@attrs.define
class MapLoader:
    tiled_map: ptp.TiledMap
    collider_handler: ColliderHandler = attrs.Factory(ColliderHandler)

    def find_source(self, gid: int) -> ptp.Tileset:
        gid &= ~(
            HORIZONTAL_FLIP_FLAG
            | DIAGONAL_FLIP_FLAG
            | VERTICAL_FLIP_FLAG
            | HEXAGONAL_ROTATION_FLAG
        )
        for tileset in self.tiled_map.tilesets.values():
            if 0 <= gid - tileset.firstgid < tileset.tile_count:
                return tileset
        raise ValueError(f"Could not find tileset for tile {gid!r}")

    def load_tile_colliders(self) -> None:
        for tileset in self.tiled_map.tilesets.values():
            self.collider_handler.load_colliders(tileset)

    def place_object(self, tiled_object: ptp_obj.TiledObject) -> p3d.NodePath:
        object_path = p3d.NodePath(tiled_object.name)
        if isinstance(tiled_object, ptp_obj.Tile):
            tileset = self.find_source(tiled_object.gid)
            tile_arranger = TileArranger(tileset)
            tile_arranger.add_tile(tiled_object.gid, 0, 0)
            tile = tile_arranger.generate_node()
            tile_path = object_path.attach_new_node(tile)
            object_width, object_height = tiled_object.size
            width_scale = object_width / self.tiled_map.tile_size.width
            height_scale = object_height / self.tiled_map.tile_size.height
            tile_path.set_pos(0, 0, object_height)
            tile_path.set_scale(width_scale, 1, height_scale)
            collision_node = self.collider_handler.get_collider(tiled_object.gid)
            if collision_node is not None:
                tile_path.attach_new_node(collision_node)
        object_path.set_pos(tiled_object.coordinates.x, 0, -tiled_object.coordinates.y)
        object_path.set_r(tiled_object.rotation)
        return object_path

    def load_layer(self, layer: ptp.Layer) -> p3d.NodePath:
        if isinstance(layer, ptp.TileLayer):
            return self.load_tile_layer(layer)
        elif isinstance(layer, ptp.ObjectLayer):
            return self.load_object_layer(layer)
        elif isinstance(layer, ptp.ImageLayer):
            layer.image = self.tiled_map.map_file.parent / layer.image
            return load_image_layer(layer)
        elif isinstance(layer, ptp.LayerGroup):
            return self.load_layer_group(layer)
        else:
            raise UnsupportedError(f"Unsupported layer type: {type(layer)!r}")

    def load_tile_layer(self, layer: ptp.TileLayer) -> p3d.NodePath:
        tile_width, tile_height = self.tiled_map.tile_size
        layer_node = p3d.NodePath(layer.name)
        collider_parent = p3d.NodePath('cluster_collider')
        tile_arranger = TileArranger(self.tiled_map.tilesets[1])
        for j, row in enumerate(layer.data or ()):
            for i, gid in enumerate(row):
                tile_arranger.add_tile(gid, i, j)
                collision_node = self.collider_handler.get_collider(gid)
                if collision_node is not None:
                    collider_path = collider_parent.attach_new_node('collider_location')
                    collider_path.set_pos(i * tile_width, 0, -j * tile_height)
                    collider_path.attach_new_node(collision_node)
        cluster_node = tile_arranger.generate_node()
        cluster_path = p3d.NodePath(cluster_node)
        collider_parent.reparent_to(cluster_path)
        cluster_path.reparent_to(layer_node)
        return layer_node

    def load_object_layer(self, layer: ptp.ObjectLayer) -> p3d.NodePath:
        layer_node = p3d.NodePath(layer.name)
        for tiled_object in layer.tiled_objects:
            object_node = self.place_object(tiled_object)
            object_node.reparent_to(layer_node)
        return layer_node

    def load_layer_group(self, layer: ptp.LayerGroup) -> p3d.NodePath:
        layer_node = p3d.NodePath(layer.name)
        for sublayer in layer.layers or ():
            sublayer_node = self.load_layer(sublayer)
            sublayer_node.reparent_to(layer_node)
        return layer_node

    def load_map(self) -> p3d.NodePath:
        map_root = p3d.NodePath(self.tiled_map.map_file.stem)
        for layer in self.tiled_map.layers:
            layer_node = self.load_layer(layer)
            layer_node.reparent_to(map_root)
        return map_root


def load_map(path: pathlib.Path) -> p3d.NodePath:
    tiled_map = ptp.parse_map(path)
    if tiled_map.infinite:
        raise UnsupportedError("Cannot load an infinite map")
    if tiled_map.orientation != 'orthogonal':
        raise UnsupportedError(
            f"Unsupported map orientation: {tiled_map.orientation!r}"
        )
    map_loader = MapLoader(tiled_map)
    map_loader.load_tile_colliders()
    return map_loader.load_map()
