// MongoDB 初始化脚本：创建应用数据库和集合及索引

db = db.getSiblingDB(process.env.MONGO_INITDB_DATABASE || 'agribot');

// ---- sessions 集合 ----
db.createCollection('sessions');
db.sessions.createIndex({ session_id: 1 }, { unique: true });
db.sessions.createIndex({ user_id: 1, updated_at: -1 });
db.sessions.createIndex({ user_id: 1, is_active: 1 });

// ---- messages 集合 ----
db.createCollection('messages');
db.messages.createIndex({ session_id: 1, created_at: 1 });
db.messages.createIndex({ created_at: 1 });

print('=== agribot MongoDB init done ===');
