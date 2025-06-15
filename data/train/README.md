# Amsterdam Osdpm Area Counterfactual Routing Dataset

This dataset provides geographic data in the Osdpm area of Amsterdam.

---

## Dataset Structure

The dataset consists of one comprehensive map file and four segmented map files:

- `osdpm_map.gpkg` — Full map of the Osdpm area.
- `osdpm_segment_1.gpkg`  
- `osdpm_segment_2.gpkg`  
- `osdpm_segment_3.gpkg`  
- `osdpm_segment_4.gpkg`  

> Each segmented file represents a specific geographic portion of the full map. 
---

## Metadata

Each foil route is associated with user preferences and map configurations. These are captured in a metadata JSON structure used for routing models.

### Example Metadata

```json
{
  "user_model": {
    "max_curb_height": 0.04,
    "min_sidewalk_width": 0.8,
    "walk_bike_preference": "walk",
    "attrs_variable_names": [
      "path_type",
      "curb_height_max",
      "obstacle_free_width_float"
    ],
    "crossing_weight_factor": 1.4,
    "walk_bike_preference_weight_factor": 0.6,
    "route_error_threshold": 0.05
  },
  "map": {
    "map_name": "osdpm_segment_1.gpkg",
    "CRS": "EPSG:28992",
    "CRS_map": "EPSG:4326",
    "area_choice": "wijken",
    "my_areas": []
  }
}
```
### user_model

| Field                             | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `max_curb_height`                | Maximum tolerable curb height (in meters).                                   |
| `min_sidewalk_width`             | Minimum sidewalk width considered walkable (in meters).                    |
| `walk_bike_preference`           | Preferred mode  — `"walk"` or `"bike"`.                           |
| `attrs_variable_names`           | List of attribute.               |
| `crossing_weight_factor`         | Penalty multiplier for crossing streets; higher values discourage crossings.|
| `walk_bike_preference_weight_factor` | Weight factor that prioritizes the user's travel mode preference.      |
| `route_error_threshold`          | Acceptable route error.        |

#### Map

| Field         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `map_name`    | Name of the corresponding map file (e.g., `osdpm_segment_1.gpkg`). |
| `CRS`         | Local projected coordinate reference system (e.g., `EPSG:28992`).           |
| `CRS_map`     | Geographic coordinate system used for mapping (e.g., `EPSG:4326`).          |
| `area_choice` | Method of regional subdivision (e.g., `"wijken"` for neighborhoods).        |
| `my_areas`    | Custom-defined areas or spatial filters, if any.                            |
