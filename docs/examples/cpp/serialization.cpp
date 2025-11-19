// Serialization usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
// Serialize to JSON object
auto json_data = object.to_json();

// Deserialize from JSON object
auto object_from_json = ObjectType::from_json(json_data);

// Serialize to JSON file
object.to_json_file("filename.ext.json");  // Extension depends on object type

// Deserialize from JSON file
auto object_from_json_file = ObjectType::from_json_file("filename.ext.json");
// end-cell-1

// start-cell-2
// Serialize to HDF5 file
object.to_hdf5_file("filename.ext.h5");  // Extension depends on object type

// Deserialize from HDF5 file
auto object_from_hdf5_file = ObjectType::from_hdf5_file("filename.ext.h5");
// end-cell-2
