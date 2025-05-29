#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/options.h"
#include "rocksdb/utilities/checkpoint.h"
#include "test_util/testharness.h"

using namespace rocksdb;

constexpr int kNumColumnFamilies = 16;
constexpr int kNumWriterThreads = 4;
constexpr int kNumExporterThreads = 10;
constexpr int kNumIterations = 100;

void TestConcurrentCheckpointExport() {
  std::string db_path = "/tmp/rocksdb_checkpoint_test";
  std::string export_base_path = "/tmp/rocksdb_checkpoint_exports";

  Env::Default()->CreateDir(export_base_path);

  DB* db;
  Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;
  options.info_log_level = InfoLogLevel::DEBUG_LEVEL;

  std::vector<ColumnFamilyDescriptor> cf_descriptors;
  cf_descriptors.push_back(
      ColumnFamilyDescriptor(kDefaultColumnFamilyName, ColumnFamilyOptions()));

  for (int i = 0; i < kNumColumnFamilies; i++) {
    cf_descriptors.push_back(ColumnFamilyDescriptor("cf_" + std::to_string(i),
                                                    ColumnFamilyOptions()));
  }

  std::vector<ColumnFamilyHandle*> cf_handles;
  EXPECT_OK(DB::Open(options, db_path, cf_descriptors, &cf_handles, &db));

  WriteOptions write_opts;
  write_opts.disableWAL = true;
  for (int i = 0; i < kNumColumnFamilies; i++) {
    ColumnFamilyHandle* cf = cf_handles[i + 1];
    std::string key = "key_" + std::to_string(i);
    EXPECT_OK(db->Put(write_opts, cf, key, std::string("0")));
  }

  std::atomic<uint64_t> put_counter(1);
  std::atomic<uint64_t> export_counter(1);
  std::atomic<int> empty_exports(0);
  std::atomic<int> total_exports(0);
  std::atomic<int> value_mismatches(0);
  std::atomic<bool> running(true);

  std::atomic<int> next_writer_cf(0);
  std::atomic<int> next_exporter_cf(0);

  auto writer = [&](int _) {
    WriteOptions write_opts;
    write_opts.disableWAL = true;
    FlushOptions flush_options;
    flush_options.wait = false;

    while (running.load()) {
      int cf_index = next_writer_cf.fetch_add(1) % kNumColumnFamilies;

      ColumnFamilyHandle* cf = cf_handles[cf_index + 1];  // skip default CF

      std::string key = "key_" + std::to_string(cf_index);
      std::string value = std::to_string(put_counter.fetch_add(1));
      EXPECT_OK(db->Put(write_opts, cf, key, value));
      EXPECT_OK(db->Flush(flush_options, cf));
    }
  };

  auto checkpoint_exporter = [&](int thread_id) {
    ReadOptions read_opts;
    uint64_t iter;

    while (true) {
      iter = export_counter.fetch_add(1);
      if (iter > kNumColumnFamilies * kNumIterations) {
        std::cout << "Terminating after " << iter << " iterations" << std::endl;
        break;
      }

      int cf_index = next_exporter_cf.fetch_add(1) % kNumColumnFamilies;

      ColumnFamilyHandle* cf =
          cf_handles[cf_index + 1];  // skip over default CF
      std::string cf_name = "cf_" + std::to_string(cf_index);
      std::string key = "key_" + std::to_string(cf_index);

      std::string value;
      uint64_t value_in_db = 0;
      Status s = db->Get(read_opts, cf, key, &value);
      if (s.ok()) {
        value_in_db = std::stoull(value);
      }

      Checkpoint* checkpoint;
      EXPECT_OK(Checkpoint::Create(db, &checkpoint));

      std::string export_path =
          export_base_path + "/" + cf_name + "_" + std::to_string(iter);
      ExportImportFilesMetaData* metadata = nullptr;

      EXPECT_OK(checkpoint->ExportColumnFamily(cf, export_path, &metadata));
      total_exports++;

      if (metadata->files.empty()) {
        empty_exports++;

        ColumnFamilyMetaData cf_metadata;
        db->GetColumnFamilyMetaData(cf, &cf_metadata);

        std::cout << "Thread " << thread_id << " CF " << cf_index
                  << ": Empty export! CF has " << cf_metadata.file_count
                  << " files after export" << std::endl;
      } else {
        std::string temp_cf_name = cf_name + "_import_" + std::to_string(iter) +
                                   "_t" + std::to_string(thread_id);
        ColumnFamilyHandle* import_cf_handle = nullptr;
        ImportColumnFamilyOptions import_options;
        import_options.move_files = false;

        EXPECT_OK(db->CreateColumnFamilyWithImport(
            ColumnFamilyOptions(), temp_cf_name, import_options, *metadata,
            &import_cf_handle));

        std::string read_value;
        s = db->Get(read_opts, import_cf_handle, key, &read_value);
        uint64_t value_in_export = std::stoull(read_value);
        if ((s.ok() && value_in_export < value_in_db) || s.IsNotFound()) {
          std::cout << "Mismatch from re-imported checkpoint for key '" << key
                    << "': expected >= '" << value_in_db << "' but got: '"
                    << value_in_export << "' in checkpoint: " << export_path
                    << std::endl;
          value_mismatches++;

          // preserve the DB/CF without further modifications
          exit(-1);
          // break;
        } else if (!s.ok() && !s.IsNotFound()) {
          std::cout << "Error from DB::Get for key '" << key
                    << "': " << s.ToString()
                    << "' in checkpoint: " << export_path << std::endl;
        } else {
          EXPECT_OK(db->DropColumnFamily(import_cf_handle));
        }

        delete import_cf_handle;
      }

      delete metadata;
      delete checkpoint;

      if (thread_id == 0) {
        std::cout << "Iterations completed: " << total_exports << std::endl;
      }
    }
  };

  std::vector<std::thread> exporters;
  for (int i = 0; i < kNumExporterThreads; i++) {
    exporters.emplace_back(checkpoint_exporter, i);
  }

  std::vector<std::thread> writers;
  for (int i = 0; i < kNumWriterThreads; i++) {
    writers.emplace_back(writer, i);
  }

  // When the exporter tasks are finished, we can terminate
  for (auto& t : exporters) {
    t.join();
  }

  running.store(false);
  for (auto& t : writers) {
    t.join();
  }

  std::cout << "\n=== Test Results ===" << std::endl;
  std::cout << "Total exports: " << total_exports << std::endl;
  std::cout << "Empty exports: " << empty_exports << std::endl;
  std::cout << "Column families: " << kNumColumnFamilies << std::endl;
  std::cout << "Writer threads: " << kNumWriterThreads << std::endl;
  std::cout << "Exporter threads: " << kNumExporterThreads << std::endl;

  for (auto* handle : cf_handles) {
    delete handle;
  }
  delete db;

  EXPECT_EQ(empty_exports, 0);
  EXPECT_EQ(value_mismatches, 0);

  // EXPECT_OK(DestroyDB(db_path, Options()));
}

int main() {
  std::cout << "Testing concurrent checkpoint exports..." << std::endl;
  TestConcurrentCheckpointExport();

  return 0;
}
